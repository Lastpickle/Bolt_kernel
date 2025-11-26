/*
 * blu_schedutil - Battery-tuned schedutil-based cpufreq governor (PELT-safe)
 *
 * Copyright (C) 2018-2025 (adapted)
 * Author: adapted for MTK PELT by @BystanderforU
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2.
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/cpufreq.h>
#include <linux/kthread.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/errno.h>
#include <linux/mutex.h>
#include <linux/irq_work.h>
#include <linux/kobject.h>
#include <linux/sysfs.h>
#include <linux/uaccess.h>
#include <linux/ctype.h>
#include <linux/jiffies.h>
#include <linux/delay.h>

#include <trace/events/power.h>

/* Fallback stubs so backported trees compile with no fast-switch support */
#ifndef cpufreq_driver_fast_switch
#define cpufreq_driver_fast_switch(policy, freq) CPUFREQ_ENTRY_INVALID
#endif
#ifndef cpufreq_enable_fast_switch
#define cpufreq_enable_fast_switch(policy)
#endif
#ifndef cpufreq_disable_fast_switch
#define cpufreq_disable_fast_switch(policy)
#endif

/* Constants tuned for battery (conservative) */
#define DEFAULT_UP_RATE_US   2000 /* 2 ms scaled by transition latency */
#define DEFAULT_DOWN_RATE_US 2000
#define DEFAULT_HISPEED_LOAD 95
#define DEFAULT_STALE_MS     50   /* 50ms window for stale util */

struct blu_tunables {
	/* simple tunables exposed via sysfs fallback */
	unsigned int up_rate_limit_us;
	unsigned int down_rate_limit_us;
	unsigned int hispeed_load;
	unsigned int hispeed_freq;
	bool iowait_boost_enable;
	unsigned int boost; /* simple boost amount (percent of capacity) */
};

struct blu_policy {
	struct cpufreq_policy *policy;
	struct blu_tunables *tunables;

	raw_spinlock_t update_lock; /* protect next_freq, last_update */
	u64 last_freq_update_time;
	u64 last_cyc_update_time;
	s64 up_rate_delay_ns;
	s64 down_rate_delay_ns;
	s64 min_rate_limit_ns;
	unsigned int next_freq;
	unsigned int cached_raw_freq;
	unsigned long max; /* capacity */
	unsigned long hispeed_util;
	unsigned long curr_cycles;

	/* slow path work */
	struct irq_work irq_work;
	struct mutex work_lock;
	struct kthread_worker worker;
	struct kthread_work work;
	struct task_struct *thread;
	bool work_in_progress;
	bool need_freq_update;
};

struct blu_cpu {
	struct update_util_data update_util;
	struct blu_policy *bp;
	unsigned long util;
	unsigned long max;
	unsigned int cpu;
	unsigned long iowait_boost;
	unsigned long iowait_boost_max;
	u64 last_update;
	/* flags reserved for simplified use */
	unsigned int flags;
};

static DEFINE_PER_CPU(struct blu_cpu, blu_cpu);
static DEFINE_PER_CPU(struct blu_tunables *, cached_tunables);

/* minimal global tunables fallback via /sys/kernel/blu_schedutil/ */
static unsigned int blu_boost = 0;
static unsigned int blu_up_rate = DEFAULT_UP_RATE_US;
static unsigned int blu_down_rate = DEFAULT_DOWN_RATE_US;
static struct kobject *blu_kobj;

static ssize_t blu_show_attr(struct kobject *kobj, struct kobj_attribute *attr,
			     char *buf)
{
	if (!attr || !attr->attr.name)
		return -EINVAL;

	if (strcmp(attr->attr.name, "boost") == 0)
		return sprintf(buf, "%u\n", blu_boost);
	if (strcmp(attr->attr.name, "up_rate") == 0)
		return sprintf(buf, "%u\n", blu_up_rate);
	if (strcmp(attr->attr.name, "down_rate") == 0)
		return sprintf(buf, "%u\n", blu_down_rate);

	return -EINVAL;
}

static ssize_t blu_store_attr(struct kobject *kobj, struct kobj_attribute *attr,
			      const char *buf, size_t count)
{
	unsigned long val;
	int ret;

	ret = kstrtoul(buf, 10, &val);
	if (ret)
		return ret;

	if (strcmp(attr->attr.name, "boost") == 0) {
		blu_boost = (unsigned int)val;
		return count;
	}
	if (strcmp(attr->attr.name, "up_rate") == 0) {
		blu_up_rate = (unsigned int)val;
		return count;
	}
	if (strcmp(attr->attr.name, "down_rate") == 0) {
		blu_down_rate = (unsigned int)val;
		return count;
	}

	return -EINVAL;
}

static struct kobj_attribute blu_attr_boost  = __ATTR(boost, 0664, blu_show_attr, blu_store_attr);
static struct kobj_attribute blu_attr_uprate = __ATTR(up_rate, 0664, blu_show_attr, blu_store_attr);
static struct kobj_attribute blu_attr_downrate = __ATTR(down_rate, 0664, blu_show_attr, blu_store_attr);

static struct attribute *blu_attrs[] = {
	&blu_attr_boost.attr,
	&blu_attr_uprate.attr,
	&blu_attr_downrate.attr,
	NULL,
};

static struct attribute_group blu_attr_group = {
	.attrs = blu_attrs,
};

static int __init blu_sysfs_init(void)
{
	int ret;

	blu_kobj = kobject_create_and_add("blu_schedutil", kernel_kobj);
	if (!blu_kobj)
		return -ENOMEM;

	ret = sysfs_create_group(blu_kobj, &blu_attr_group);
	if (ret) {
		kobject_put(blu_kobj);
		blu_kobj = NULL;
		return ret;
	}
	return 0;
}

static void blu_sysfs_exit(void)
{
	if (blu_kobj) {
		sysfs_remove_group(blu_kobj, &blu_attr_group);
		kobject_put(blu_kobj);
		blu_kobj = NULL;
	}
}

/* helper: simple freq->util conversion */
static unsigned long freq_to_util(unsigned long max_cap, unsigned int freq, unsigned int max_freq)
{
	if (!max_freq)
		return 0;
	return mult_frac(max_cap, freq, max_freq);
}

/* compute desired next frequency */
static unsigned int blu_get_next_freq(struct blu_policy *bp, unsigned long util, unsigned long max)
{
	struct cpufreq_policy *policy = bp->policy;
	unsigned int base_freq;

	/* choose base frequency: frequency-invariant assumption disabled for battery */
	base_freq = policy->cur;

	/* Conservative scaling: use a smaller multiplier to favor lower freq */
	/* next_freq = 1.05 * base_freq * util / max */
	base_freq = (base_freq + (base_freq >> 4)); /* +6.25% */
	if (!max)
		max = 1;

	base_freq = mult_frac(base_freq, util, max);

	/* avoid returning same raw freq repeatedly */
	if (base_freq == bp->cached_raw_freq && bp->next_freq != UINT_MAX)
		return bp->next_freq;
	bp->cached_raw_freq = base_freq;

	return cpufreq_driver_resolve_freq(policy, base_freq);
}

/* simple stale detection: return window in ns */
static u64 blu_stale_ns(void)
{
	return (u64)DEFAULT_STALE_MS * NSEC_PER_MSEC;
}

/* update util for single CPU */
static void blu_update_single(struct update_util_data *hook, u64 time, unsigned int flags)
{
	struct blu_cpu *bcpu = container_of(hook, struct blu_cpu, update_util);
	struct blu_policy *bp = bcpu->bp;
	struct cpufreq_policy *policy = bp->policy;
	unsigned long util, max;
	unsigned int next_f;
	bool should;

	/* Keep iowait boosting simple */
	if (bp->tunables && bp->tunables->iowait_boost_enable) {
		if (flags & SCHED_CPUFREQ_IOWAIT)
			bcpu->iowait_boost = bcpu->iowait_boost_max;
		else if (bcpu->iowait_boost) {
			u64 delta = time - bcpu->last_update;
			if (delta > (u64)TICK_NSEC)
				bcpu->iowait_boost = 0;
		}
	}

	bcpu->last_update = time;

	/* simple rate limiting check */
	if (bp->need_freq_update) {
		bp->need_freq_update = false;
		bp->next_freq = UINT_MAX;
	} else {
		if (time - bp->last_freq_update_time < bp->min_rate_limit_ns)
			return;
	}

	/* gather util: use rq pelt avg or fallback */
	util = min((unsigned long)cpu_rq(bcpu->cpu)->cfs.avg.util_avg, arch_scale_cpu_capacity(NULL, bcpu->cpu));
	max = arch_scale_cpu_capacity(NULL, bcpu->cpu);

	/* apply tiny boost if requested (global) */
	if (blu_boost)
		util = max(util, mult_frac(max, blu_boost, 100));

	/* apply local iowait boost */
	if (bcpu->iowait_boost)
		util = max(util, bcpu->iowait_boost);

	/* compute next freq conservatively */
	next_f = blu_get_next_freq(bp, util, max);

	/* rate-limit up/down */
	{
		s64 delta_ns = time - bp->last_freq_update_time;
		if (next_f > bp->next_freq && delta_ns < bp->up_rate_delay_ns) {
			/* too soon to increase */
			bp->cached_raw_freq = 0;
			return;
		}
		if (next_f < bp->next_freq && delta_ns < bp->down_rate_delay_ns) {
			/* too soon to decrease */
			bp->cached_raw_freq = 0;
			return;
		}
	}

	/* commit */
	if (bp->next_freq == next_f)
		return;

	bp->next_freq = next_f;
	bp->last_freq_update_time = time;

	if (policy->fast_switch_enabled) {
		unsigned int retf = cpufreq_driver_fast_switch(policy, next_f);
		if (retf == CPUFREQ_ENTRY_INVALID)
			return;
		policy->cur = retf;
	} else {
		bp->work_in_progress = true;
		irq_work_queue(&bp->irq_work);
	}
}

/* scheduled work for slow path (do the actual cpufreq target) */
static void blu_work_fn(struct kthread_work *work)
{
	struct blu_policy *bp = container_of(work, struct blu_policy, work);

	mutex_lock(&bp->work_lock);
	__cpufreq_driver_target(bp->policy, bp->next_freq, CPUFREQ_RELATION_L);
	mutex_unlock(&bp->work_lock);

	bp->work_in_progress = false;
}

/* irq work handler */
static void blu_irq_work(struct irq_work *irq_work)
{
	struct blu_policy *bp = container_of(irq_work, struct blu_policy, irq_work);

	kthread_queue_work(&bp->worker, &bp->work);
}

/* allocate/free policy */
static struct blu_policy *blu_policy_alloc(struct cpufreq_policy *policy)
{
	struct blu_policy *bp;

	bp = kzalloc(sizeof(*bp), GFP_KERNEL);
	if (!bp)
		return NULL;

	bp->policy = policy;
	raw_spin_lock_init(&bp->update_lock);
	return bp;
}
static void blu_policy_free(struct blu_policy *bp)
{
	kfree(bp);
}

/* create kthread worker if needed (slow path) */
static int blu_kthread_create(struct blu_policy *bp)
{
	struct task_struct *thread;
	struct cpufreq_policy *policy = bp->policy;
	int cpu_first = cpumask_first(policy->related_cpus);
	int ret = 0;

	if (policy->fast_switch_enabled)
		return 0;

	kthread_init_work(&bp->work, blu_work_fn);
	kthread_init_worker(&bp->worker);

	thread = kthread_create(kthread_worker_fn, &bp->worker, "blu_sched:%d", cpu_first);
	if (IS_ERR(thread)) {
		pr_err("blu_schedutil: kthread creation failed\n");
		return PTR_ERR(thread);
	}

	bp->thread = thread;
	init_irq_work(&bp->irq_work, blu_irq_work);
	mutex_init(&bp->work_lock);

	wake_up_process(thread);
	return ret;
}

static void blu_kthread_stop(struct blu_policy *bp)
{
	if (bp->policy->fast_switch_enabled)
		return;

	kthread_flush_worker(&bp->worker);
	kthread_stop(bp->thread);
	mutex_destroy(&bp->work_lock);
}

/* tunables management (simple caching per-cpu if required) */
static struct blu_tunables *blu_tunables_alloc(struct blu_policy *bp)
{
	struct blu_tunables *tun;

	tun = kzalloc(sizeof(*tun), GFP_KERNEL);
	if (!tun)
		return NULL;

	/* defaults favor battery */
	tun->up_rate_limit_us = blu_up_rate;
	tun->down_rate_limit_us = blu_down_rate;
	tun->hispeed_load = DEFAULT_HISPEED_LOAD;
	tun->hispeed_freq = 0;
	tun->iowait_boost_enable = false;
	tun->boost = 0;

	return tun;
}
static void blu_tunables_free(struct blu_tunables *tun)
{
	kfree(tun);
}

/* sugov-like init */
static int blu_init(struct cpufreq_policy *policy)
{
	struct blu_policy *bp;
	struct blu_tunables *tun;
	unsigned int lat;
	int ret;

	if (policy->governor_data)
		return -EBUSY;

	/* enable fast-switch hint (may be no-op) */
	cpufreq_enable_fast_switch(policy);

	bp = blu_policy_alloc(policy);
	if (!bp) {
		ret = -ENOMEM;
		goto disable_fast_switch;
	}

	ret = blu_kthread_create(bp);
	if (ret)
		goto free_bp;

	/* alloc tunables */
	tun = blu_tunables_alloc(bp);
	if (!tun) {
		ret = -ENOMEM;
		goto stop_kthread;
	}

	tun->up_rate_limit_us = blu_up_rate;
	tun->down_rate_limit_us = blu_down_rate;

	/* scale by policy latency if present */
	lat = policy->cpuinfo.transition_latency / NSEC_PER_USEC;
	if (lat) {
		tun->up_rate_limit_us *= lat;
		tun->down_rate_limit_us *= lat;
	}

	policy->governor_data = bp;
	bp->tunables = tun;

	/* set rate limit ns */
	bp->up_rate_delay_ns = (s64)tun->up_rate_limit_us * NSEC_PER_USEC;
	bp->down_rate_delay_ns = (s64)tun->down_rate_limit_us * NSEC_PER_USEC;
	bp->min_rate_limit_ns = min(bp->up_rate_delay_ns, bp->down_rate_delay_ns);

	bp->last_freq_update_time = 0;
	bp->next_freq = UINT_MAX;
	bp->cached_raw_freq = 0;
	bp->work_in_progress = false;
	bp->need_freq_update = false;
	bp->hispeed_util = 0;

	/* init CPUs in policy */
	{
		unsigned int cpu;
		for_each_cpu(cpu, policy->cpus) {
			struct blu_cpu *bcpu = &per_cpu(blu_cpu, cpu);
			memset(bcpu, 0, sizeof(*bcpu));
			bcpu->bp = bp;
			bcpu->cpu = cpu;
			bcpu->iowait_boost_max = policy->cpuinfo.max_freq;
			/* register hook */
			cpufreq_add_update_util_hook(cpu, &bcpu->update_util, policy_is_shared(policy) ? blu_update_single : blu_update_single);
		}
	}

	return 0;

stop_kthread:
	blu_kthread_stop(bp);
free_bp:
	blu_policy_free(bp);
disable_fast_switch:
	cpufreq_disable_fast_switch(policy);
	pr_err("blu_schedutil: init failed %d\n", ret);
	return ret;
}

static void blu_exit(struct cpufreq_policy *policy)
{
	struct blu_policy *bp = policy->governor_data;
	unsigned int cpu;

	if (!bp)
		return;

	for_each_cpu(cpu, policy->cpus)
		cpufreq_remove_update_util_hook(cpu);

	cpufreq_disable_fast_switch(policy);

	blu_kthread_stop(bp);

	blu_tunables_free(bp->tunables);
	policy->governor_data = NULL;
	blu_policy_free(bp);
}

/* start/stop/limits */
static int blu_start(struct cpufreq_policy *policy)
{
	struct blu_policy *bp = policy->governor_data;
	unsigned int cpu;

	bp->up_rate_delay_ns = (s64)bp->tunables->up_rate_limit_us * NSEC_PER_USEC;
	bp->down_rate_delay_ns = (s64)bp->tunables->down_rate_limit_us * NSEC_PER_USEC;
	bp->min_rate_limit_ns = min(bp->up_rate_delay_ns, bp->down_rate_delay_ns);
	bp->last_freq_update_time = 0;
	bp->next_freq = UINT_MAX;
	bp->work_in_progress = false;
	bp->need_freq_update = false;
	bp->cached_raw_freq = 0;

	for_each_cpu(cpu, policy->cpus) {
		struct blu_cpu *bcpu = &per_cpu(blu_cpu, cpu);
		memset(bcpu, 0, sizeof(*bcpu));
		bcpu->bp = bp;
		bcpu->cpu = cpu;
		bcpu->iowait_boost_max = policy->cpuinfo.max_freq;
		/* re-register hooks */
		cpufreq_add_update_util_hook(cpu, &bcpu->update_util, blu_update_single);
	}

	return 0;
}

static void blu_stop(struct cpufreq_policy *policy)
{
	struct blu_policy *bp = policy->governor_data;
	unsigned int cpu;

	for_each_cpu(cpu, policy->cpus)
		cpufreq_remove_update_util_hook(cpu);

	synchronize_sched();

	if (!policy->fast_switch_enabled) {
		irq_work_sync(&bp->irq_work);
		kthread_cancel_work_sync(&bp->work);
	}
}

/* limit callback */
static void blu_limits(struct cpufreq_policy *policy)
{
	struct blu_policy *bp = policy->governor_data;
	unsigned long flags;

	if (!policy->fast_switch_enabled) {
		mutex_lock(&bp->work_lock);
		raw_spin_lock_irqsave(&bp->update_lock, flags);
		/* no cycle tracking in PELT fallbacks */
