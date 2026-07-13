# Taskmaster Planning Defaults v0.2 组会汇报摘要

## 当前状态

当前版本已经完成生成、模型预审核、定点修复、修复后复审和合并。

- 数据域：`planning_defaults`
- 源数据：Taskmaster-2 `flights` + `hotels`
- 生成方式：基于 Taskmaster seed 场景，用 `gpt-5.5` + `reasoning_effort=xhigh` 逐条生成 session/probe
- 用户数：30
- session 数：1080
- probe 数：120
- 每个用户：36 个 session + 4 个 probe
- 每个用户 probe 类型：`direct_use`、`boundary`、`exception`、`explicit_retrieval` 各 1 个
- 全局 habit 类型数：12
- 最终模型审核：120 accept / 0 revise / 0 reject
- 合并校验：通过，未发现 ID 对齐、gold 缺失、evidence 缺失、禁用词泄漏等结构问题

## 长对话长度

- 平均 session 长度：3331.2 字符
- 中位 session 长度：3276 字符
- P10 session 长度：2732 字符
- 最短 session：1824 字符
- 最长 session：5277 字符
- 平均 messages/session：19.0
- 中位 messages/session：18
- 最少 messages/session：12

## 12 类 Habit

1. `business_travel_arrival_buffer`
   - 商务/客户 onsite 出行时，倾向选择有充分到达缓冲的 itinerary，而不是更便宜但卡点的安排。

2. `tight_schedule_nonstop_priority`
   - 日程紧、同日会议、延误敏感时，倾向选择 nonstop 或低转机方案，而不是便宜但多转机方案。

3. `uncertain_trip_refundable_fare`
   - 工作行程未完全确定、会议/审批可能变化时，倾向选择 refundable/flexible fare。

4. `work_hotel_near_venue`
   - 工作、会议、客户拜访住宿时，倾向住在靠近会议地点的酒店，而不是更便宜但通勤远的酒店。

5. `quiet_hotel_for_work`
   - 商务出行或需要准备材料/开会时，倾向选择安静、有可靠 workspace 的酒店。

6. `leisure_relaxed_pacing`
   - 休闲/周末/低压力旅行时，倾向更放松的节奏，而不是最大化每个小时。

7. `family_trip_flexible_cancellation`
   - 家庭出行、带孩子或多人同行时，倾向选择 flexible cancellation/change policy。

8. `short_trip_no_checked_bag`
   - 一两晚短途或快速工作访问时，倾向 carry-on friendly，避免需要 checked baggage 的方案。

9. `international_long_layover_buffer`
   - 国际航班、涉及海关/入境/换机场时，倾向更长 layover buffer。

10. `red_eye_avoidance`
    - 可选情况下倾向避免 red-eye，除非价格或日程优势很大。

11. `airport_transfer_reliability`
    - 客户/会议/工作出行需要机场地面交通时，倾向可预测、可靠的 transfer。

12. `early_morning_avoidance`
    - 非紧急或灵活出行时，倾向避免 very early departure。

## 评测设计

每个用户有 4 个 probe：

- `direct_use`：当前请求在 habit 适用范围内，模型应使用历史中归纳出的 planning default。
- `boundary`：当前请求不在 habit 适用范围内，模型不应过度套用历史偏好。
- `exception`：当前请求显式给出 override，模型应服从当前约束，而不是机械使用默认偏好。
- `explicit_retrieval`：显式问用户历史中反复体现的 planning preference，用作记忆检索 sanity check。

## 审核与修复流程

1. 初始生成后，120 个 probe 通过自动结构校验。
2. 用 `gpt-5.5 xhigh` 做全量模型预审核：
   - 101 accept
   - 19 revise
3. 19 条 revise 的问题主要分为四类：
   - API/解析失败导致的标注失败
   - distractor 太弱，正确项过于明显
   - probe 证据或关键细节不够贴合
   - 正确项不够唯一，和其他选项相比优势不明显
4. 对 19 条做定点修复，只改 probe 层，不改 session。
5. 修复后再次用 `gpt-5.5 xhigh` 审核：
   - 19 accept
   - 0 revise
   - 0 reject
6. 修复结果已合并进主文件，旧主文件已归档到 `archive/pre_repair19_merge_20260710_213906`。

## 表面线索控制

在人工检查样例时发现一个额外问题：部分 distractor 更常带 `but`、`though`、`despite` 等负向转折词，而 gold 选项更干净。这可能让模型不用读历史，仅根据选项语气排除 distractor。

修复前统计：

- Gold choices 带 contrast marker：14 / 120 = 11.7%
- Distractor choices 带 contrast marker：74 / 360 = 20.6%
- 高风险 probe：17-21 条，主要集中在 `boundary` 和 `direct_use`

修复策略：

- 只修复 surface-cue 高风险 probe，不重生成 session。
- 目标不是删除所有 tradeoff，而是让 gold 和 distractor 的语气结构更均衡。
- 避免“错误项都带明显 negative caveat，正确项没有 caveat”的模式。

修复后统计：

- Gold choices 带 contrast marker：21 / 120 = 17.5%
- Distractor choices 带 contrast marker：48 / 360 = 13.3%
- Gold choices broad caveat rate：27.5%
- Distractor choices broad caveat rate：19.4%
- 高风险 contrast-marker probe：0
- 高风险 broad-caveat probe：0
- 修复后的 21 条 probe 经过 `gpt-5.5 xhigh` 复审：21 accept / 0 revise / 0 reject

这一步已经合并进主文件，旧主文件归档到 `archive/pre_surface_cue_merge_20260712_112946`。

## 代表样例

### Hidden Habit

- User: `tm2_planning_v02_user_0004`
- Habit: `quiet_hotel_for_work`
- 条件：business travel 或需要准备材料、开会、集中工作的 trip
- 默认行为：优先选择安静、有可靠 workspace 的酒店，而不是 nightlife-heavy 或 amenity-focused 酒店
- boundary：社交/休闲旅行，希望 nightlife 和娱乐
- exception：当前请求明确要求 lively atmosphere 或 group entertainment

### Public Probe

Query:

> I'm booking a three-night hotel stay in Seattle for a client workshop. I'll be in sessions most of the day, with some time in the evenings to catch up on emails and deck edits. Which option should I choose?

Choices:

- A: A hotel attached to the workshop venue, with the shortest commute, a popular lobby bar, compact rooms, and desk space that reviewers say is better for quick emails than longer work blocks.
- B: A newer hotel in a restaurant-heavy area with a rooftop lounge, strong design reviews, and easy access to dinner spots, though guests often mention a lively evening atmosphere.
- C: A full-service property a short rideshare from the workshop, with a gym, pool, breakfast package, and business center, but mixed room reviews for Wi-Fi strength and work surfaces.
- D: A smaller business-oriented hotel on an office-heavy block, about a 10-minute walk from the workshop, with reliable Wi-Fi, full in-room desks, and fewer on-site amenities or nightlife options.

Gold:

- `D`

原因：

该用户历史中多次在 work-trip hotel 场景表达“安静、可靠 Wi-Fi、真实 desk、能晚上工作”优先于 nightlife/amenities。D 不是在所有普通酒店指标上绝对占优，但最符合该用户的 scoped planning default。

### Evidence Session 摘要

Evidence 1: `tm2_planning_v02_user_0004_s0016`

- 用户要去 Seattle 参加 client workshop。
- 用户明确说需要 check in、unpack、review deck，不想为了省一点钱牺牲准备时间。
- 私有 evidence summary：用户在 client workshop trip 中优先考虑 quiet hotel、workspace 和睡眠，而不是 lively/amenity-focused property。

关键原文：

```text
user: I need help lining up a Delta trip to Seattle for next month. It is for a client workshop, so I also need a hotel that will let me get real work done.

user: The client is near South Lake Union. I do not need a fancy scene. I need a quiet room, dependable Wi-Fi, and somewhere I can sit with my laptop for a few hours.

assistant: I see three plausible choices: a business-oriented hotel in South Lake Union with good desk space, a boutique waterfront hotel with a popular bar and music at night, and a larger downtown hotel with many amenities but a busier lobby. Which direction sounds right?

user: Please lean toward the business-oriented place. The waterfront one sounds fun for a weekend, but not when I need to rehearse and sleep before presenting.

user: Breakfast would be nice, but it is less important than a calm place to work. I can expense coffee; I cannot expense a bad night of sleep before the workshop.

assistant: Got it. I would recommend the early Delta nonstop to Seattle, the Thursday afternoon nonstop back to Atlanta, and the quieter South Lake Union hotel with reliable workspace rather than the livelier waterfront or amenity-heavy downtown choices.
```

Evidence 2: `tm2_planning_v02_user_0004_s0017`

- 用户要为 Seattle work trip 选酒店。
- 用户说不想住在 loud 地方，需要 decent Wi-Fi、real desk、quiet enough for evening calls。
- 用户明确排除了 rooftop bar 型酒店，认为 fun for another trip but not this one。

关键原文：

```text
user: Hi, I need help picking a hotel in Seattle for a work trip next month.

user: June 11 to 13, just me. I have vendor meetings near the Convention Center and I’ll need to finish a slide deck at night. I’d rather not end up somewhere loud.

user: Walking distance would be nice, but not if it means staying above bars or a clubby lobby. I need decent Wi-Fi, a real desk, and quiet enough for a couple of evening calls.

assistant: I found three plausible options: The Paramount Hotel, close to the Convention Center with business-friendly rooms; Hotel Theodore, stylish and central but busier in the lobby; and a boutique property near Pike Place with a rooftop bar and smaller rooms.

user: The rooftop bar one sounds fun for another trip, but not this one. Between the first two, which is better for working after meetings?

assistant: The Paramount looks stronger for your needs. It is close enough to walk, tends to be calmer than nightlife-focused hotels, and rooms commonly have proper desks. Hotel Theodore is convenient too, but the lobby and common areas are more social.

user: That matches what I’m looking for. I don’t care about a scene or fancy amenities if I can sleep and prepare.

user: No breakfast requirement. If there’s coffee nearby I’m fine. Quiet workspace matters more than breakfast or a gym for this trip.
```

## 当前结论

当前版本可以作为 planning_defaults 数据域的高质量候选 benchmark slice：

- 数据规模清晰，30 users / 1080 sessions / 120 probes。
- 长对话长度明显高于之前短 session 版本。
- 全局 habit 类型为 12，满足“10-20 种 habit 类型”的设计目标。
- 每个 probe 都有 private key 和 evidence session ids。
- 修复后全量模型复审为 120 accept。
- 表面线索审计后，高风险 contrast/caveat probe 已降为 0。
- 当前仍建议在正式 release 前做一轮人工 spot check，尤其看 repaired 19 条是否符合人类直觉。
