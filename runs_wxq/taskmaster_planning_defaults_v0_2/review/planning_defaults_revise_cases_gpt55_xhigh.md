# GPT-5.5 xhigh Revise Cases

Source: `planning_defaults_review_queue_all_model_labeled_gpt55_xhigh.csv`

Total revise cases: **19**

## Overview

| # | Probe Type | User | Habit Template | Reason |
|---:|---|---|---|---|
| 1 | `boundary` | `tm2_planning_v02_user_0002` | `uncertain_trip_refundable_fare` | revise: The boundary gold is correct, but the query and distractor wording make the lowest-fare answer too obvious rather than ... |
| 2 | `boundary` | `tm2_planning_v02_user_0009` | `red_eye_avoidance` | revise: The hidden scope and gold are coherent, but the query and distractors make B trivial by stating overnight travel is the... |
| 3 | `boundary` | `tm2_planning_v02_user_0011` | `early_morning_avoidance` | revise: The boundary setup is coherent and gold C is supported, but the choices make C nearly trivial because all alternatives ... |
| 4 | `boundary` | `tm2_planning_v02_user_0023` | `early_morning_avoidance` | revise: The boundary scenario and gold are coherent, but the choices explicitly say the distractors violate approval or miss th... |
| 5 | `boundary` | `tm2_planning_v02_user_0027` | `work_hotel_near_venue` | revise: The boundary logic and gold are supported, but the query/choice B make the non-work office option too obviously irrelev... |
| 6 | `boundary` | `tm2_planning_v02_user_0029` | `leisure_relaxed_pacing` | revise: The boundary scope and gold are coherent, but the distractors make A obvious on ordinary fixed-schedule travel grounds ... |
| 7 | `direct_use` | `tm2_planning_v02_user_0004` | `quiet_hotel_for_work` | revise: The in-scope habit and gold are coherent, but option D is objectively dominant while the distractors have clear work-tr... |
| 8 | `direct_use` | `tm2_planning_v02_user_0006` | `family_trip_flexible_cancellation` | revise: The sample is coherent and in-scope, but the query's explicit moving parts around schedules makes the flexible-cancella... |
| 9 | `direct_use` | `tm2_planning_v02_user_0012` | `business_travel_arrival_buffer` | revise: Choice D also provides a prior-night arrival buffer for a 1 p.m. workshop, so the proposed gold is not uniquely determi... |
| 10 | `direct_use` | `tm2_planning_v02_user_0013` | `tight_schedule_nonstop_priority` | revise: Choice B is an ambiguous alternate-airport option rather than a clearly higher-transfer itinerary, so the gold is less ... |
| 11 | `direct_use` | `tm2_planning_v02_user_0017` | `leisure_relaxed_pacing` | revise: model labeling failed; manually review (RuntimeError) |
| 12 | `direct_use` | `tm2_planning_v02_user_0021` | `red_eye_avoidance` | revise: The red-eye avoidance habit is supported and in scope, but the options need cleanup because B contradicts the arrival c... |
| 13 | `direct_use` | `tm2_planning_v02_user_0024` | `business_travel_arrival_buffer` | revise: Gold D is broadly supported, but B also lands the prior evening with a plausible overnight buffer, making the scoped de... |
| 14 | `direct_use` | `tm2_planning_v02_user_0028` | `quiet_hotel_for_work` | revise: B is uniquely correct under the graph, but the evidence preview only shows work-flight timing and preparation needs rat... |
| 15 | `exception` | `tm2_planning_v02_user_0001` | `tight_schedule_nonstop_priority` | revise: The exception query and gold are coherent, but the supplied evidence preview is hotel-focused and does not substantiate... |
| 16 | `exception` | `tm2_planning_v02_user_0007` | `short_trip_no_checked_bag` | revise: The exception gold is correct and supported, but the distractors—especially the option explicitly saying to ship items ... |
| 17 | `explicit_retrieval` | `tm2_planning_v02_user_0026` | `uncertain_trip_refundable_fare` | revise: model labeling failed; manually review (RuntimeError) |
| 18 | `explicit_retrieval` | `tm2_planning_v02_user_0027` | `work_hotel_near_venue` | revise: model labeling failed; manually review (RuntimeError) |
| 19 | `explicit_retrieval` | `tm2_planning_v02_user_0028` | `quiet_hotel_for_work` | revise: The gold matches the hidden default, but the shown support evidence is flight-planning/prep context rather than lodging... |

## Details

### 1. tm2_planning_v02_user_0002_habit_uncertain_trip_refundable_fare_p01_boundary

- Probe type: `boundary`
- User: `tm2_planning_v02_user_0002`
- Public probe id: `taskmaster_planning_v02_probe_9e59fed95619d968`
- Proposed gold: `A` / `do_not_apply_out_of_scope`
- Model decision: `revise`

Reviewer note:

> revise: The boundary gold is correct, but the query and distractor wording make the lowest-fare
> answer too obvious rather than testing scoped non-application of the flexible-fare default.

Hidden habit graph:

- Template: `uncertain_trip_refundable_fare`
- Name: uncertain trips prefer refundable fares
- Condition: work trips with tentative meetings, pending approvals, or plans that may shift
- Default action: Prefer refundable or flexible fares even when a nonrefundable fare is cheaper.
- Boundary condition: fixed personal trips with confirmed dates
- Exception condition: current request explicitly says the dates are locked and lowest fare is the priority

Query:

A user is booking a personal trip to Madison for a family reunion. The reunion dates are fixed,
their time off is already approved, and they say their top priority is keeping the airfare as low as
possible. Which fare should you recommend?

Choices:

- `A` **[gold]**: Book the lowest-priced nonrefundable economy fare that matches the fixed dates.
- `B`: Choose a more expensive flexible fare with no change fees, even though the dates are confirmed.
- `C`: Pay extra for a fully refundable fare in case the reunion schedule changes.
- `D`: Select a mid-priced fare with a travel credit option, despite the user prioritizing the lowest
       upfront cost.

Evidence preview:

- `tm2_planning_v02_user_0002_s0001` | index `1` | signal `boundary_counterexample` | source `hotels`
  - user: Hi, can you help me find a hotel in Madison, Wisconsin?
  - assistant: Sure. What dates, how many guests, and do you want to be near downtown, the
    university, or somewhere with easier parking?
  - user: It's a personal trip for my niece's graduation weekend. The dates are confirmed: check in
    Friday May 16 and check out Monday May 19. Two adults, one room, ki...
  - assistant: Got it. For graduation weekend, rooms near campus can fill quickly. Will you have a
    car, and is walking distance to campus important?
  - user: We'll drive from Chicago, so parking matters. We'd like to be close enough to walk to
    State Street or campus if the weather is decent. Budget target is under...
- `tm2_planning_v02_user_0002_s0021` | index `21` | signal `boundary_counterexample` | source `hotels`
  - user: Hi, I need help picking a hotel in Palm Springs, California.
  - assistant: Sure, I can help with that. What dates are you staying, and how many guests will
    there be?
  - user: Two adults, May 16 to May 19. It’s for our anniversary. Flights into PSP are already
    booked and we have tickets for a show Saturday night, so the dates are set.
  - assistant: Got it: three nights, two adults, fixed anniversary weekend. What matters most for
    the hotel: price, pool, walkability, parking, resort feel, or something qu...
  - user: A pool is important, and I’d like to be close enough to downtown that we can Uber cheaply
    or walk to dinner. I don’t want a huge party hotel. King bed if pos...

### 2. tm2_planning_v02_user_0009_habit_red_eye_avoidance_p01_boundary

- Probe type: `boundary`
- User: `tm2_planning_v02_user_0009`
- Public probe id: `taskmaster_planning_v02_probe_b63be0a2857d5ab1`
- Proposed gold: `B` / `do_not_apply_out_of_scope`
- Model decision: `revise`

Reviewer note:

> revise: The hidden scope and gold are coherent, but the query and distractors make B trivial by
> stating overnight travel is the only way to meet a hard work constraint while alternatives absurdly
> risk missing work.

Hidden habit graph:

- Template: `red_eye_avoidance`
- Name: avoid red-eye unless savings are large
- Condition: flight planning where an overnight flight is optional
- Default action: Avoid red-eye flights unless the price or schedule advantage is substantial.
- Boundary condition: trips where overnight travel is necessary or the user asks to preserve daytime hours
- Exception condition: current request explicitly asks for red-eye timing or maximum daytime availability

Query:

A traveler is flying from Honolulu to Boston after attending a Sunday evening family event that ends
at 9 p.m. They need to be back at work Tuesday morning, and the only itineraries that arrive in time
depart late Sunday night and fly overnight. Which planning recommendation best fits this situation?

Choices:

- `A`: Avoid all overnight options and recommend a Monday daytime departure, even though it would risk
       missing work Tuesday morning.
- `B` **[gold]**: Include the late-night overnight options and compare them on arrival time, connection quality, and
                  price, since overnight travel is needed to meet the Tuesday-morning constraint.
- `C`: Suggest staying an extra night in Honolulu and asking to miss Tuesday morning work so the flight can
       be during the day.
- `D`: Choose a much more expensive Monday daytime itinerary by default because avoiding overnight travel
       should take priority over the work timing.

Evidence preview:

- `tm2_planning_v02_user_0009_s0008` | index `8` | signal `boundary_counterexample` | source `flights`
  - user: I’m trying to get to Los Angeles and need help narrowing down flights.
  - assistant: Sure. What city are you leaving from, what dates are you considering, and do you have
    a latest arrival time in Los Angeles?
  - user: Leaving Honolulu. I have to be at a meeting near LAX by 8:30 Monday morning, but I cannot
    leave before 8:15 Sunday night because of a family dinner. I want t...
  - assistant: That timing will likely mean an overnight flight to LAX. Is that acceptable for this
    trip? Also, how many travelers, and will you check a bag?
  - user: Just me, one checked bag plus a backpack. And yes, this is one of the rare times an
    overnight is okay. Normally I avoid red-eyes unless the savings are huge...
- `tm2_planning_v02_user_0009_s0032` | index `32` | signal `boundary_counterexample` | source `flights`
  - user: Hi, I’m arranging a round trip from Los Angeles to Paris for a design review week. I want
    to get the flights settled first.
  - assistant: Sure. What dates are you traveling, and are you leaving from LAX or another Los
    Angeles-area airport?
  - user: LAX is best. I need to leave Friday, September 13, and come back Sunday, September 22. For
    my usual trips I try not to take overnight flights unless the fare...
  - assistant: Got it. For this Paris trip, do you still want me to avoid overnight timing, or is
    there a scheduling reason to consider it?
  - user: This one is different. I have meetings in LA until about 5:30 that Friday, and I don’t
    want to burn the whole workday. So leaving at night is fine here, even...

### 3. tm2_planning_v02_user_0011_habit_early_morning_avoidance_p01_boundary

- Probe type: `boundary`
- User: `tm2_planning_v02_user_0011`
- Public probe id: `taskmaster_planning_v02_probe_ce26e5597497d8c5`
- Proposed gold: `C` / `do_not_apply_out_of_scope`
- Model decision: `revise`

Reviewer note:

> revise: The boundary setup is coherent and gold C is supported, but the choices make C nearly
> trivial because all alternatives are late or very risky rather than plausible ways to trade off the
> early-flight default.

Hidden habit graph:

- Template: `early_morning_avoidance`
- Name: avoid very early departures for nonurgent trips
- Condition: nonurgent personal or flexible travel where departure time is optional
- Default action: Prefer mid-morning or later departures over very early flights when tradeoffs are modest.
- Boundary condition: business or event travel where arriving early is critical
- Exception condition: current request explicitly needs the earliest possible arrival

Query:

A user needs to fly to New York for a mandatory 10:00 AM client kickoff, says being late would be
unacceptable, and cannot travel the night before. Which outbound flight should you recommend?

Choices:

- `A`: 9:30 AM nonstop, arriving at 11:45 AM, for the lowest fare.
- `B`: 10:15 AM nonstop with extra legroom, arriving at 12:35 PM, for a moderate fare.
- `C` **[gold]**: 6:00 AM nonstop, arriving at 8:15 AM, for $80 more than the cheapest option.
- `D`: 7:45 AM one-stop flight, arriving at 9:50 AM if the connection is on time, for $30 less.

Evidence preview:

- `tm2_planning_v02_user_0011_s0003` | index `3` | signal `boundary_counterexample` | source `hotels`
  - user: I need help finding a hotel in New York for next Wednesday night.
  - assistant: Sure. What area of New York would you like to be near, and what is your budget? Also,
    are you only looking for the hotel, or do you need help with getting th...
  - user: This is for a trade media breakfast at the Javits Center on Thursday morning. I'm coming
    from Cleveland, and I can't fly in the night before because of a boa...
  - assistant: Got it. Since you need to be at Javits Thursday morning, the inbound flight is
    important. What time do you need to be on site, and do you have a hotel budget...
  - user: Tech check is at 8:30 AM sharp, and they asked presenters to be there before attendees
    arrive. Hotel under $350 before taxes if possible, quiet room, decent...
- `tm2_planning_v02_user_0011_s0034` | index `34` | signal `boundary_counterexample` | source `flights`
  - user: I need help choosing a round trip from Seattle to Salt Lake City. Outbound is today,
    September 13, and I come back September 18. This is for a client kickoff...
  - assistant: Understood. For the outbound, arrival time matters more than comfort. Are you flying
    from Sea-Tac, carrying bags, and do you have any airline preference?
  - user: Sea-Tac, carry-on only, and I have PreCheck. Delta or Alaska are both fine. I know the
    first flight is painful, but missing the opening session would be worse.
  - assistant: I found three plausible nonstop options: Delta at 5:40 AM arriving SLC 8:35 AM for
    about $328, Alaska at 7:25 AM arriving 10:18 AM for about $287, and Delta...
  - user: The 7:25 is tempting since it is cheaper and less miserable. How tight would that be if I
    take a rideshare from SLC?

### 4. tm2_planning_v02_user_0023_habit_early_morning_avoidance_p01_boundary

- Probe type: `boundary`
- User: `tm2_planning_v02_user_0023`
- Public probe id: `taskmaster_planning_v02_probe_068cf2d835763cdb`
- Proposed gold: `C` / `do_not_apply_out_of_scope`
- Model decision: `revise`

Reviewer note:

> revise: The boundary scenario and gold are coherent, but the choices explicitly say the distractors
> violate approval or miss the sign-in, making the answer too trivial.

Hidden habit graph:

- Template: `early_morning_avoidance`
- Name: avoid very early departures for nonurgent trips
- Condition: nonurgent personal or flexible travel where departure time is optional
- Default action: Prefer mid-morning or later departures over very early flights when tradeoffs are modest.
- Boundary condition: business or event travel where arriving early is critical
- Exception condition: current request explicitly needs the earliest possible arrival

Query:

A traveler is booking same-day business travel from Sacramento to Portland for a client kickoff. The
client requires in-person sign-in at 9:00 a.m.; the office is about 25 minutes from PDX, and the
company will not approve an overnight stay because a same-day option can work. Which flight
recommendation best fits the situation?

Choices:

- `A`: Book a flight the previous evening and add a hotel night; it avoids a dawn departure, but violates
       the no-overnight approval rule.
- `B`: Book the 8:30 a.m. nonstop arriving at 10:05 a.m.; it is more comfortable and $45 cheaper, but
       misses the required sign-in.
- `C` **[gold]**: Book the 5:50 a.m. nonstop arriving at 7:25 a.m.; it leaves enough time to reach the client before
                  sign-in, even though it is a dawn departure.
- `D`: Book the 10:15 a.m. one-stop arriving at 1:20 p.m.; it is the lowest fare and avoids the early
       start, but misses the kickoff.

Evidence preview:

- `tm2_planning_v02_user_0023_s0000` | index `0` | signal `boundary_counterexample` | source `flights`
  - user: I need help sorting out flights for two trips from Oakland. One is a casual weekend in
    Phoenix next month, and the other is a work day in Portland.
  - assistant: Sure. Let's handle them one at a time. What are the dates for the Phoenix weekend,
    and how flexible are you on timing?
  - user: Phoenix is April 12 to April 14, visiting my sister. Totally flexible. For trips like
    that, I really don't want a dawn departure if I can avoid it. I'd rathe...
  - assistant: For Oakland to Phoenix, I would skip the 6:10 a.m. nonstop even though it is about
    $38 cheaper. A 10:35 a.m. nonstop gets you in around 12:25 p.m. and keeps...
  - user: That sounds like the right style for Phoenix. Put that one aside as my likely choice.
- `tm2_planning_v02_user_0023_s0018` | index `18` | signal `boundary_counterexample` | source `flights`
  - user: Can you help me sort out a few upcoming trips? I’m starting with flights, and I’d rather
    not make bad choices just because something is a little cheaper.
  - assistant: Sure. Tell me the first trip: origin, destination, dates, and whether your timing is
    flexible.
  - user: First one is a personal weekend in Portland next month. I can leave Friday anytime, but I
    really don’t want a dawn flight if it only saves twenty or thirty d...
  - assistant: That makes sense. For a flexible weekend, I’d focus on late morning or afternoon
    departures unless the price difference is substantial. Do you want nonstop o...
  - user: Nonstop preferred. Last time I took a 5:35 a.m. flight for a casual trip, I was useless
    the whole first day.

### 5. tm2_planning_v02_user_0027_habit_work_hotel_near_venue_p01_boundary

- Probe type: `boundary`
- User: `tm2_planning_v02_user_0027`
- Public probe id: `taskmaster_planning_v02_probe_1b8bbb7f3f718e19`
- Proposed gold: `A` / `do_not_apply_out_of_scope`
- Model decision: `revise`

Reviewer note:

> revise: The boundary logic and gold are supported, but the query/choice B make the non-work office
> option too obviously irrelevant rather than a plausible travel-planning distractor.

Hidden habit graph:

- Template: `work_hotel_near_venue`
- Name: work hotels prefer venue proximity
- Condition: work, conference, client, or onsite meeting hotel selection
- Default action: Prefer hotels close to the meeting venue over cheaper hotels with a longer commute.
- Boundary condition: leisure trips where neighborhood character or sightseeing access matters more
- Exception condition: current request explicitly prioritizes budget or a specific neighborhood over commute time

Query:

A traveler is planning a leisure birthday weekend in New York City. They say they care most about
staying in a neighborhood with character, good restaurants and bars, and easy access to sightseeing.
They are not attending any meetings, and being near their company’s Midtown office is irrelevant.
Which hotel recommendation best fits?

Choices:

- `A` **[gold]**: A boutique hotel in the West Village or SoHo, with strong restaurant options nearby and convenient
                  subway access to major sights, even if it costs a bit more than a business hotel farther uptown.
- `B`: A business-oriented hotel next to the Midtown office, because it minimizes the commute to a place
       associated with their job even though they will not be going there.
- `C`: The cheapest hotel near JFK Airport, accepting a long ride into Manhattan each day in exchange for
       the lowest nightly rate.
- `D`: A large Times Square hotel with tour-bus pickup at the door, despite heavy crowds and a less
       distinctive neighborhood feel.

Evidence preview:

- `tm2_planning_v02_user_0027_s0011` | index `11` | signal `boundary_counterexample` | source `hotels`
  - user: I'm planning a New York City stay in October for a long weekend and could use help
    choosing a hotel.
  - assistant: Sure. What are your dates, how many people are traveling, and is this more of a
    sightseeing trip or tied to anything scheduled?
  - user: October 10 to 14, two adults. It's an anniversary trip, not work, and we're not tied to
    any conference hall or office. My partner hasn't really spent time in...
  - assistant: That helps. Do you have a nightly budget before taxes? Also, any preferences around
    room size, noise, transit, or neighborhoods you already like or want to a...
  - user: I'd like to stay around $320 before taxes, but I can stretch to about $370 if the area is
    noticeably better. Queen bed is fine. Quiet matters, elevator matte...
- `tm2_planning_v02_user_0027_s0013` | index `13` | signal `boundary_counterexample` | source `hotels`
  - user: Hello, could you help me find a hotel in San Francisco for a long weekend?
  - assistant: Absolutely. What dates are you looking at, how many guests, and is this more of a
    work stay or a leisure trip?
  - user: Two adults, May 16 through May 19. It is purely for fun. My partner has never been to San
    Francisco, so I want it to feel like a proper city weekend, not jus...
  - assistant: Great. Do you have a nightly budget and any neighborhoods you already like or want to
    avoid?
  - user: Ideally under $300 before taxes. I would rather pay a little more for a neighborhood with
    character, cafes, walkable streets, and easy access to sightseeing....

### 6. tm2_planning_v02_user_0029_habit_leisure_relaxed_pacing_p01_boundary

- Probe type: `boundary`
- User: `tm2_planning_v02_user_0029`
- Public probe id: `taskmaster_planning_v02_probe_087c00372f3b8a74`
- Proposed gold: `A` / `do_not_apply_out_of_scope`
- Model decision: `revise`

Reviewer note:

> revise: The boundary scope and gold are coherent, but the distractors make A obvious on ordinary
> fixed-schedule travel grounds rather than isolating the scoped leisure-pacing default.

Hidden habit graph:

- Template: `leisure_relaxed_pacing`
- Name: leisure trips prefer relaxed pacing
- Condition: vacation, weekend, or low-pressure personal travel planning
- Default action: Prefer relaxed arrival times and fewer tightly packed logistics over maximizing every hour.
- Boundary condition: work trips or event travel with fixed commitments
- Exception condition: current request explicitly asks to maximize sightseeing time or fit a hard schedule

Query:

You are arranging a San Diego work trip for a client workshop. The traveler must attend a prep
dinner Sunday at 7:00 PM, facilitate sessions Monday 9:00 AM–5:00 PM, and join a debrief Tuesday
8:30–10:00 AM. Which itinerary approach is best?

Choices:

- `A` **[gold]**: Book a Sunday morning nonstop that arrives by early afternoon, choose a hotel within a short ride of
                  the workshop site, and schedule the return after the Tuesday debrief with extra airport buffer.
- `B`: Fly in Saturday on the earliest flight and pack Sunday with sightseeing before the dinner, even if
       it means a tighter pre-workshop schedule.
- `C`: Book a later Sunday flight for an easier start, keep hotel options open until arrival, and plan to
       reach the prep dinner if timing works out.
- `D`: Choose the cheapest connection arriving Sunday evening shortly before dinner, then use rideshares as
       needed to recover time if there are delays.

Evidence preview:

- `tm2_planning_v02_user_0029_s0004` | index `4` | signal `boundary_counterexample` | source `flights`
  - user: I need to get to San Diego for a client workshop, not a vacation. Can you help me sort out
    flights and where I should stay?
  - assistant: Sure. Which city are you flying from, what dates, and where in San Diego is the
    workshop? San Diego International is usually the practical airport for downto...
  - user: Flying from Seattle. I need to be at a training center in Mission Valley by 8:15 a.m.
    Tuesday, April 16. The workshop runs Tuesday through Thursday, and Thur...
  - assistant: Given the 8:15 a.m. Tuesday commitment, I would not plan a same-day arrival. For
    Mission Valley, use San Diego International and stay near the training cente...
  - user: I have to be in the Bellevue office until about 2:15 p.m. Monday. I’ll have TSA PreCheck,
    one carry-on, and a checked tube with demo posters.
- `tm2_planning_v02_user_0029_s0024` | index `24` | signal `boundary_counterexample` | source `flights`
  - user: I need help finding a flight to London, England.
  - assistant: Sure. What city are you leaving from, and what dates are you considering?
  - user: From Austin. It is for work, not vacation. I need to be in London for a client kickoff
    Monday at 9:00, and there is a prep dinner Sunday at 7:30 near Bloomsb...
  - assistant: Got it. So arrival on Sunday with enough margin before dinner is important. When can
    you leave Austin, and when do you want to return?
  - user: I can leave Saturday anytime after 10 a.m. Return Thursday after 5 p.m. London time.
    Economy is fine, company cap is about $1600, and I will check one bag.

### 7. tm2_planning_v02_user_0004_habit_quiet_hotel_for_work_p00_direct_use

- Probe type: `direct_use`
- User: `tm2_planning_v02_user_0004`
- Public probe id: `taskmaster_planning_v02_probe_e9111e4887239b12`
- Proposed gold: `D` / `apply_scoped_habit`
- Model decision: `revise`

Reviewer note:

> revise: The in-scope habit and gold are coherent, but option D is objectively dominant while the
> distractors have clear work-trip drawbacks, making the answer too trivial without relying on the
> learned default.

Hidden habit graph:

- Template: `quiet_hotel_for_work`
- Name: work hotels prefer quiet properties
- Condition: business travel or trips requiring preparation, calls, or focused work
- Default action: Prefer quieter hotels with reliable workspace over nightlife-heavy or amenity-focused properties.
- Boundary condition: social or leisure trips where nightlife and amenities are the goal
- Exception condition: current request explicitly asks for lively atmosphere or group entertainment

Query:

I'm booking a hotel for a three-night trip to Seattle for a client workshop. I arrive the evening
before and may need to prepare materials between sessions. Which option should I choose?

Choices:

- `A`: A lower-priced airport hotel with free shuttle service, but a long commute to the workshop and
       limited workspace in the rooms.
- `B`: A stylish hotel in the main nightlife district with a rooftop bar, late-night restaurant, and
       energetic lobby scene.
- `C`: A resort-style property with a pool, spa, and several restaurants, but mixed comments about room
       noise and internet consistency.
- `D` **[gold]**: A well-reviewed hotel near the workshop site with dependable Wi-Fi, a proper in-room desk, and rooms
                  away from busy bar streets.

Evidence preview:

- `tm2_planning_v02_user_0004_s0016` | index `16` | signal `support` | source `flights`
  - user: I need help lining up a Delta trip to Seattle for next month. It is for a client workshop,
    so I also need a hotel that will let me get real work done.
  - assistant: Sure. What city are you flying from, what dates, and do you have any timing
    constraints for the flights?
  - user: Leaving from Atlanta on Monday, May 13, returning Thursday, May 16. I need to be in
    Seattle by midafternoon Monday because I have prep calls that evening.
  - assistant: I can look for Delta options arriving before midafternoon. Do you prefer nonstop if
    available, or are you open to one connection for a lower fare?
  - user: Nonstop if it is not wildly more expensive. For this trip the schedule matters more than
    shaving off a little money.
- `tm2_planning_v02_user_0004_s0017` | index `17` | signal `support` | source `hotels`
  - user: Hi, I need help picking a hotel in Seattle for a work trip next month.
  - assistant: Sure. What dates, budget, and part of Seattle are you considering?
  - user: June 11 to 13, just me. I have vendor meetings near the Convention Center and I’ll need to
    finish a slide deck at night. I’d rather not end up somewhere loud.
  - assistant: Got it. Do you want to be walking distance to the Convention Center, or is a short
    rideshare okay?
  - user: Walking distance would be nice, but not if it means staying above bars or a clubby lobby.
    I need decent Wi-Fi, a real desk, and quiet enough for a couple of...
- `tm2_planning_v02_user_0004_s0018` | index `18` | signal `support` | source `flights`
  - user: I need to arrange a round trip flight from Austin to Seattle for a client workshop next
    month, and probably a hotel too.
  - assistant: Sure. What dates are you traveling, and do you have any time or airline preferences?
  - user: Leave Monday, May 13, and come back Thursday evening, May 16. I need to be in Bellevue by
    2 pm Monday, so a morning arrival would be best.
  - assistant: I found a few reasonable round-trip options. The strongest fit is a 7:10 am departure
    from Austin with one stop in Denver, landing in Seattle at 11:35 am. Re...
  - user: Let’s keep the buffer. I’ll have materials to review when I land, so I don’t want to be
    rushing straight into the room.

### 8. tm2_planning_v02_user_0006_habit_family_trip_flexible_cancellation_p00_direct_use

- Probe type: `direct_use`
- User: `tm2_planning_v02_user_0006`
- Public probe id: `taskmaster_planning_v02_probe_ff5b2ee42921199d`
- Proposed gold: `C` / `apply_scoped_habit`
- Model decision: `revise`

Reviewer note:

> revise: The sample is coherent and in-scope, but the query's explicit moving parts around schedules
> makes the flexible-cancellation answer overly obvious without relying on the learned default.

Hidden habit graph:

- Template: `family_trip_flexible_cancellation`
- Name: family trips prefer flexible cancellation
- Condition: family travel, trips with children, or plans involving multiple relatives
- Default action: Prefer flexible cancellation and change policies over the absolute cheapest prepaid option.
- Boundary condition: solo fixed-date trips with low uncertainty
- Exception condition: current request explicitly says plans are locked and budget is the only priority

Query:

I'm booking a four-night Chicago hotel stay for my spouse, our two kids, and my mother. The dates
line up with school break, but there are still a few moving parts around activities and my mother's
schedule. Which option would you recommend?

Choices:

- `A`: Pick a cheaper hotel farther from the main sights and plan to use rideshares to offset the savings.
- `B`: Reserve the largest suite available, even though it is much more expensive, to maximize space for
       everyone.
- `C` **[gold]**: Choose a slightly more expensive hotel with free cancellation until a few days before arrival and no
                  change fee.
- `D`: Book the lowest nightly rate: a prepaid, nonrefundable room at a well-rated hotel near the river.

Evidence preview:

- `tm2_planning_v02_user_0006_s0009` | index `9` | signal `support` | source `hotels`
  - user: Hi, I need help finding a hotel in San Francisco for a family weekend in late April.
  - assistant: Sure. What dates, how many people, and what part of the city are you hoping to stay
    in?
  - user: Friday April 25 to Monday April 28. It’ll be me, my husband, our two kids, and my parents.
    We’ll need either two rooms or a suite plus an extra room.
  - assistant: Got it. Are you planning to focus on sightseeing, visiting family, or a specific
    event?
  - user: Mostly sightseeing: Exploratorium, ferry building, maybe Golden Gate Park. My parents
    don’t want steep walks every time we leave the hotel.
- `tm2_planning_v02_user_0006_s0015` | index `15` | signal `support` | source `hotels`
  - user: I’m putting together a Palm Springs weekend for my family in late April and need help
    narrowing down hotels.
  - assistant: Sure. What dates, how many people, and any must-haves for the hotel?
  - user: Friday April 26 to Monday April 29. Two adults, our two kids, and possibly my
    mother-in-law if she decides to join. So I’m looking at either a suite or conne...
  - assistant: Got it. Are you aiming for a resort stay, something close to downtown Palm Springs,
    or the lowest nightly price?
  - user: Pool matters more than being downtown. The kids will want to swim. Also, because my
    mother-in-law may come and my son’s tournament schedule can shift, I don’...
- `tm2_planning_v02_user_0006_s0027` | index `27` | signal `support` | source `hotels`
  - user: I'm trying to find a hotel in West Palm Beach for a long weekend in late April. It'll be
    me, my spouse, our two kids, and my parents.
  - assistant: Sure. Do you want to be closer to the beach, downtown, or near the airport? Also, how
    many rooms and what budget range should I use?
  - user: Two rooms, preferably adjoining or at least on the same floor. We'd like to stay under
    about $260 a night per room if possible. Beach access is nice, but we...
  - assistant: For a family group, I’d compare places around CityPlace/downtown and a few near Palm
    Beach Lakes Boulevard. Do you need breakfast, a pool, or any mobility co...
  - user: Breakfast and a pool would help a lot. My dad walks slowly, so I don't want a huge resort
    layout where the room is a trek from parking or the lobby.

### 9. tm2_planning_v02_user_0012_habit_business_travel_arrival_buffer_p00_direct_use

- Probe type: `direct_use`
- User: `tm2_planning_v02_user_0012`
- Public probe id: `taskmaster_planning_v02_probe_2014673fe0856b8e`
- Proposed gold: `C` / `apply_scoped_habit`
- Model decision: `revise`

Reviewer note:

> revise: Choice D also provides a prior-night arrival buffer for a 1 p.m. workshop, so the proposed
> gold is not uniquely determined by the scoped default.

Hidden habit graph:

- Template: `business_travel_arrival_buffer`
- Name: business travel prefers protected arrival buffer
- Condition: business/client/onsite meeting travel planning
- Default action: Prefer an arrival window with meaningful cushion before meeting-dependent commitments over tighter
                  cheaper schedules.
- Boundary condition: relaxed leisure or personal travel without a same-day commitment
- Exception condition: current itinerary explicitly relaxes arrival timing or moves commitments to a later day

Query:

I’m booking travel for a client onsite in Denver next Tuesday. The client workshop starts at 1:00
p.m. downtown, and I need to choose an itinerary. Which option should I pick?

Choices:

- `A`: Take a Tuesday morning connecting flight that lands at 10:50 a.m., saving about $90 but requiring a
       45-minute ride downtown.
- `B`: Take the cheapest Tuesday morning flight, land at 11:45 a.m., and use a lower-cost hotel 30 minutes
       from the client office.
- `C` **[gold]**: Fly in Monday evening on a nonstop, stay at a hotel a short walk from the client office, and pay
                  about $160 more than the cheapest option.
- `D`: Arrive late Monday night at the airport hotel, save on the room rate, and commute downtown Tuesday
       morning during rush hour.

Evidence preview:

- `tm2_planning_v02_user_0012_s0001` | index `1` | signal `support` | source `hotels`
  - user: Hi, I need help arranging a hotel in Madison, Wisconsin, for a client visit next month.
  - assistant: Sure. What dates will you be in Madison, and do you have a specific area or budget in
    mind?
  - user: I need to be onsite with the client Tuesday at 2:00 pm and Wednesday morning. I’ll fly in
    from Denver. Ideally downtown or near the university, but the timin...
  - assistant: Got it. Are you planning to arrive Tuesday morning, or would you consider Monday
    night?
  - user: Please check both, but I’m cautious about same-day arrivals for client meetings. If I
    arrive Tuesday, I need enough cushion to deal with delays, get to the h...
- `tm2_planning_v02_user_0012_s0007` | index `7` | signal `support` | source `hotels`
  - user: I need to set up a hotel in St. Louis for a work trip next month. It’s for a client
    onsite, so I want the timing to be clean.
  - assistant: Sure. What dates are you traveling, and where in St. Louis will the client meeting
    be?
  - user: I fly in Monday, March 18, and check out Wednesday the 20th. The client office is near
    Clayton, and the first meeting is Monday at 2:30 PM.
  - assistant: Would you prefer staying in Clayton close to the office, or somewhere downtown with
    more restaurants and nightlife?
  - user: Close to the office. This isn’t a sightseeing trip. I mainly need to be settled and not
    watching the clock before the kickoff.
- `tm2_planning_v02_user_0012_s0011` | index `11` | signal `support` | source `hotels`
  - user: Hi, I need help putting together a quick trip to Denver for a client visit next Tuesday. I
    also need a hotel room with a refrigerator because I’ll be traveli...
  - assistant: I can help with that. What time is the client visit, and do you have any preferences
    for location, budget, or hotel type?
  - user: The onsite walkthrough starts at 1:30 PM at their office near Union Station. I’ll be
    flying in from Seattle. Budget is flexible within reason, but I need thi...
  - assistant: For the hotel, staying near Union Station would make the morning easy. I can look for
    properties with in-room mini-fridges or guaranteed medical refrigeratio...
  - user: Monday night is my preference if the cost isn’t absurd. For client meetings, I don’t like
    cutting it close on arrival day. A delayed flight or a long ride fr...

### 10. tm2_planning_v02_user_0013_habit_tight_schedule_nonstop_priority_p00_direct_use

- Probe type: `direct_use`
- User: `tm2_planning_v02_user_0013`
- Public probe id: `taskmaster_planning_v02_probe_422ca21daf99334d`
- Proposed gold: `A` / `apply_scoped_habit`
- Model decision: `revise`

Reviewer note:

> revise: Choice B is an ambiguous alternate-airport option rather than a clearly higher-transfer
> itinerary, so the gold is less uniquely tied to the scoped nonstop/low-transfer default.

Hidden habit graph:

- Template: `tight_schedule_nonstop_priority`
- Name: tight schedules prefer nonstop flights
- Condition: travel with same-day meetings, short connection windows, or high delay sensitivity
- Default action: Prefer nonstop or lowest-transfer itineraries over slightly cheaper connecting options.
- Boundary condition: relaxed travel where transfers are acceptable and cost matters more
- Exception condition: current trip explicitly prioritizes budget or airline miles over schedule reliability

Query:

I’m booking an outbound flight from Denver to Seattle for next Tuesday. I land the same day as a
1:30 p.m. client kickoff downtown, and I’ll have a carry-on only. Which option should I recommend?

Choices:

- `A` **[gold]**: Take the $420 nonstop that arrives at SEA at 10:10 a.m., leaving a good buffer before the meeting.
- `B`: Take the $260 flight into Paine Field that arrives at 9:55 a.m., then plan on a longer ground
       transfer to downtown Seattle.
- `C`: Take the $290 one-stop itinerary that arrives at SEA at 12:20 p.m. with a longer layover.
- `D`: Take the $315 one-stop itinerary that arrives at SEA at 11:05 a.m. but has a 45-minute connection.

Evidence preview:

- `tm2_planning_v02_user_0013_s0001` | index `1` | signal `support` | source `hotels`
  - user: I need help setting up a quick work trip to West Palm Beach next month. It is mostly about
    the hotel, but I also need the flight to make sense.
  - assistant: Sure. What dates are you traveling, where will your meetings be, and what kind of
    hotel budget should I keep in mind?
  - user: I fly in Tuesday, May 14, and come back Wednesday night. The meeting is at a client office
    near downtown West Palm Beach, close to Clematis Street. I need to...
  - assistant: Understood. For the hotel, would you rather stay downtown within a short ride of the
    client office, or closer to Palm Beach International Airport for easier...
  - user: Downtown is better. I can take a rideshare to the airport after the Wednesday meeting. For
    Tuesday, though, the flight timing is the thing I am worried about.
- `tm2_planning_v02_user_0013_s0005` | index `5` | signal `support` | source `hotels`
  - user: I need to put together a quick work trip to Santa Fe, New Mexico. Mainly I need a hotel,
    but the flight timing may matter too.
  - assistant: Sure. What dates are you traveling, and do you want to stay near a particular part of
    Santa Fe?
  - user: I need to be there Wednesday, June 12, and check out Friday the 14th. My client meeting
    starts at 1:00 pm on Wednesday near the Plaza, so I can't have a mess...
  - assistant: Got it. For hotels near the Plaza, I can look at walkable options with
    business-friendly amenities. Are you flying in from Denver as usual, or from somewhere...
  - user: From Denver. Please look at the flight too. I know there are cheaper ways through Dallas
    or by flying into Albuquerque, but I really need to land with time t...
- `tm2_planning_v02_user_0013_s0008` | index `8` | signal `support` | source `flights`
  - user: I need help booking a flight, please.
  - assistant: Sure. What cities are you traveling between, and what dates do you have in mind?
  - user: I’m flying from Boston to Atlanta next Tuesday, returning Wednesday evening. I have a
    client meeting in Midtown at 1:30 Tuesday, so I need to land with room...
  - assistant: Got it. Do you prefer morning departure, and are there any airlines or airports you
    want to avoid?
  - user: Morning is fine, Logan to ATL. I’m not picky on airline, but I don’t want a risky
    connection. This is one of those trips where a delay would ruin the day.

### 11. tm2_planning_v02_user_0017_habit_leisure_relaxed_pacing_p00_direct_use

- Probe type: `direct_use`
- User: `tm2_planning_v02_user_0017`
- Public probe id: `taskmaster_planning_v02_probe_33209a334a4f8d31`
- Proposed gold: `D` / `apply_scoped_habit`
- Model decision: `revise`

Reviewer note:

> revise: model labeling failed; manually review (RuntimeError)

Hidden habit graph:

- Template: `leisure_relaxed_pacing`
- Name: leisure trips prefer relaxed pacing
- Condition: vacation, weekend, or low-pressure personal travel planning
- Default action: Prefer relaxed arrival times and fewer tightly packed logistics over maximizing every hour.
- Boundary condition: work trips or event travel with fixed commitments
- Exception condition: current request explicitly asks to maximize sightseeing time or fit a hard schedule

Query:

I'm planning a low-pressure three-night vacation to Charleston next month, with no fixed events
after I arrive. Which travel plan would you recommend?

Choices:

- `A`: Take the cheapest 6:00 a.m. one-stop flight, drop bags on arrival, and schedule a full afternoon of
       tours before dinner.
- `B`: Take a very early nonstop flight, rent a car immediately, and fit in a beach stop, plantation tour,
       and downtown dinner on arrival day.
- `C`: Take an evening flight after work that arrives close to midnight, saving a vacation day and starting
       sightseeing early the next morning.
- `D` **[gold]**: Take a nonstop flight that leaves midmorning and arrives early afternoon, then keep the first day to
                  hotel check-in, a short walk, and an easy dinner nearby.

Evidence preview:

- `tm2_planning_v02_user_0017_s0002` | index `2` | signal `support` | source `flights`
  - user: I'm starting to plan a family trip to Orlando from Sacramento. It's just for fun, probably
    a long weekend in early May, and I want help sorting out flights.
  - assistant: Sure. Do you have exact dates in mind, how many travelers, and any airline or timing
    preferences?
  - user: Two adults and one 9-year-old. We were thinking Friday May 3 to Monday May 6. No airline
    loyalty. Since it's a vacation, I don't want a brutal start if we ca...
  - assistant: Got it. From Sacramento to Orlando, many options connect through Phoenix, Dallas,
    Denver, or Atlanta. Would you rather prioritize lowest fare, shortest trave...
  - user: Calmer schedule, within reason. I know leaving at 5:45 in the morning might save money,
    but getting a kid out the door before sunrise is not the tone I want...
- `tm2_planning_v02_user_0017_s0007` | index `7` | signal `support` | source `hotels`
  - user: Hi, I'm planning a long weekend in New Orleans for my partner and me, and I need help
    choosing a hotel.
  - assistant: Sure, I can help with that. What dates are you considering, what nightly budget feels
    comfortable, and do you want to be in the middle of nightlife or somewh...
  - user: We're thinking Friday to Monday in late March. Budget is around $220 to $280 a night
    before taxes. This is just for fun, so I don't want the trip to feel lik...
  - assistant: That helps. For a first New Orleans weekend, the French Quarter is convenient but can
    be loud. The Central Business District and Warehouse District are close...
  - user: A quieter base sounds good. I don't mind taking a streetcar or rideshare, but I don't want
    every plan to involve crossing town at midnight. We like good food...
- `tm2_planning_v02_user_0017_s0010` | index `10` | signal `support` | source `flights`
  - user: Hi, can you help me plan a round-trip flight for a short vacation?
  - assistant: Sure. Where are you leaving from, where are you headed, and what dates are you
    considering?
  - user: My partner and I are flying from San Jose to Portland for a long weekend. Leave Friday,
    come back Monday. It’s just a relaxed getaway, no event we have to make.
  - assistant: Got it. Do you want the lowest fare, the most time in Portland, or something more
    comfortable in terms of timing?
  - user: Comfortable timing, please. I don’t want a 6 a.m. flight or anything that makes us start
    the weekend exhausted. We’d rather arrive at a normal hour and have...

### 12. tm2_planning_v02_user_0021_habit_red_eye_avoidance_p00_direct_use

- Probe type: `direct_use`
- User: `tm2_planning_v02_user_0021`
- Public probe id: `taskmaster_planning_v02_probe_1de13246272a77d7`
- Proposed gold: `C` / `apply_scoped_habit`
- Model decision: `revise`

Reviewer note:

> revise: The red-eye avoidance habit is supported and in scope, but the options need cleanup because
> B contradicts the arrival constraint and D is a plausible non-red-eye alternative, making C not
> uniquely determined.

Hidden habit graph:

- Template: `red_eye_avoidance`
- Name: avoid red-eye unless savings are large
- Condition: flight planning where an overnight flight is optional
- Default action: Avoid red-eye flights unless the price or schedule advantage is substantial.
- Boundary condition: trips where overnight travel is necessary or the user asks to preserve daytime hours
- Exception condition: current request explicitly asks for red-eye timing or maximum daytime availability

Query:

I’m booking a round-trip flight from Chicago to San Diego for a work event. I need to arrive by 2:00
PM on Tuesday, and all options below meet that. Which itinerary should you recommend?

Choices:

- `A`: An 11:45 PM nonstop arriving at 2:20 AM for $319.
- `B`: A 1:15 PM nonstop arriving at 3:50 PM for $302.
- `C` **[gold]**: A 7:30 AM nonstop arriving at 10:05 AM for $348.
- `D`: A 6:10 AM one-stop itinerary arriving at 12:45 PM for $331, with a 55-minute connection.

Evidence preview:

- `tm2_planning_v02_user_0021_s0003` | index `3` | signal `support` | source `hotels`
  - user: Hi, can you help me plan a New Orleans trip? I started out looking for a hotel, but I
    probably need flights too.
  - assistant: Sure. What dates are you considering, where are you flying from, and what kind of
    hotel area or budget would feel right?
  - user: Two adults from Seattle, May 16 to May 20. We'd like French Quarter, CBD, or Warehouse
    District, ideally walkable but not directly on the loudest block. Arou...
  - assistant: Got it. Do you have flight-time preferences, checked bags, or hotel must-haves like
    breakfast, parking, pool, or balcony?
  - user: No car, one checked bag total. Breakfast isn't important. For flights, please don't route
    us on an overnight unless it is a major win. A red-eye that only sa...
- `tm2_planning_v02_user_0021_s0004` | index `4` | signal `support` | source `flights`
  - user: Hi, I’m trying to arrange a flight from New York to Seattle for a short work trip. Can you
    help me compare a few options?
  - assistant: Of course. What dates are you traveling, and do you prefer JFK, LaGuardia, or Newark?
  - user: JFK is best. I need to leave on Tuesday, September 17, and come back Friday, September 20.
    I’ll be in meetings most of Wednesday and Thursday.
  - assistant: Got it. For JFK to Seattle, I see a few general patterns: morning nonstop flights
    that arrive around midday, afternoon nonstops that arrive early evening, an...
  - user: No strong airline preference, though Delta or Alaska would be easy. I’m okay paying a bit
    more for a normal schedule.
- `tm2_planning_v02_user_0021_s0024` | index `24` | signal `support` | source `flights`
  - user: Can you help me look at round-trip flights from Houston to Seattle? I need to leave this
    coming Sunday and come back five days later, preferably with departu...
  - assistant: Sure. To make sure I search the right airports, are you leaving from Houston
    Intercontinental, Hobby, or either? And is Seattle-Tacoma okay for arrival?
  - user: Either Houston airport is fine if the fare makes sense, but IAH is easier. Sea-Tac is the
    destination.
  - assistant: Got it. Do you have a preferred airline or any limits on stops?
  - user: One stop is okay. Nonstop is nice, but I do not need to pay a crazy premium for it.

### 13. tm2_planning_v02_user_0024_habit_business_travel_arrival_buffer_p00_direct_use

- Probe type: `direct_use`
- User: `tm2_planning_v02_user_0024`
- Public probe id: `taskmaster_planning_v02_probe_d96c4d2345d81448`
- Proposed gold: `D` / `apply_scoped_habit`
- Model decision: `revise`

Reviewer note:

> revise: Gold D is broadly supported, but B also lands the prior evening with a plausible overnight
> buffer, making the scoped default less uniquely distinguishable.

Hidden habit graph:

- Template: `business_travel_arrival_buffer`
- Name: business travel prefers protected arrival buffer
- Condition: business/client/onsite meeting travel planning
- Default action: Prefer an arrival window with meaningful cushion before meeting-dependent commitments over tighter
                  cheaper schedules.
- Boundary condition: relaxed leisure or personal travel without a same-day commitment
- Exception condition: current itinerary explicitly relaxes arrival timing or moves commitments to a later day

Query:

I need to travel from Denver to Raleigh for a client onsite meeting that starts at 9:30 a.m. on
Tuesday. Which itinerary would you recommend?

Choices:

- `A`: Take the earliest Tuesday morning flight scheduled to land at 8:05 a.m., skip the hotel Monday
       night, and go straight to the meeting.
- `B`: Fly in Monday evening landing around 7:30 p.m. and stay at a less expensive hotel near the airport,
       about 45 minutes from the client site.
- `C`: Take the cheaper Monday evening flight landing around 10:45 p.m. and stay at a midrange hotel 25
       minutes from the office.
- `D` **[gold]**: Fly in Monday afternoon, land around 3:00 p.m., and book a hotel within a 10-minute ride of the
                  client’s office, even though it costs about $140 more overall.

Evidence preview:

- `tm2_planning_v02_user_0024_s0002` | index `2` | signal `support` | source `flights`
  - user: I need help finding a flight to Houston after my friend's wedding weekend. I also have a
    client onsite there right after, so the timing matters.
  - assistant: Sure. What city are you flying from, and what day do you need to arrive in Houston?
  - user: From Cincinnati. The wedding is Saturday night, so I'd fly Sunday, June 9. I need to be in
    Houston for a client kickoff Monday at 1:00 pm near downtown.
  - assistant: Got it. Do you prefer Houston Hobby or Bush Intercontinental, and are you open to
    connections?
  - user: Either airport is okay. Hobby is a little nicer for downtown, but I don't want to
    over-optimize the airport if it makes the schedule risky. One connection is...
- `tm2_planning_v02_user_0024_s0005` | index `5` | signal `support` | source `hotels`
  - user: Hi, I need help setting up a hotel in St. Louis, Missouri, for a work trip next month.
  - assistant: Sure. What dates do you need, and do you want to stay near downtown, the airport, or
    a specific office?
  - user: Arriving Tuesday May 14 and leaving Thursday May 16. The client office is near the
    Convention Center, and I have an onsite kickoff at 3:00 pm on Tuesday.
  - assistant: Got it. Would you like me to prioritize walking distance to the Convention Center,
    lower nightly rate, or hotel amenities like breakfast and a fitness center?
  - user: Walking distance matters, but the bigger thing is not cutting my arrival close. I’ll be
    flying in that morning from Seattle, and I want enough cushion before...
- `tm2_planning_v02_user_0024_s0016` | index `16` | signal `support` | source `flights`
  - user: I need to get to Paris for a client kickoff next month. Can you help me compare flights?
  - assistant: Sure. What city are you departing from, what dates, and what time is the kickoff?
  - user: Leaving from Boston. The meeting is in La Défense on Tuesday the 18th at 3:00 pm. I can
    fly overnight Monday or very early Tuesday if it actually makes sense.
  - assistant: Got it. For BOS to Paris, I see a nonstop Monday night arriving Tuesday at 8:35 am
    into CDG for $1,240, a one-stop via Dublin arriving 11:50 am for $890, and...
  - user: Arrival timing matters more for this one. I do not want to land at 2:05 and try to make a
    3:00 client meeting after immigration and traffic.

### 14. tm2_planning_v02_user_0028_habit_quiet_hotel_for_work_p00_direct_use

- Probe type: `direct_use`
- User: `tm2_planning_v02_user_0028`
- Public probe id: `taskmaster_planning_v02_probe_009a1a9aded16857`
- Proposed gold: `B` / `apply_scoped_habit`
- Model decision: `revise`

Reviewer note:

> revise: B is uniquely correct under the graph, but the evidence preview only shows work-flight
> timing and preparation needs rather than the quiet-hotel/workspace preference, so the support
> evidence needs cleanup.

Hidden habit graph:

- Template: `quiet_hotel_for_work`
- Name: work hotels prefer quiet properties
- Condition: business travel or trips requiring preparation, calls, or focused work
- Default action: Prefer quieter hotels with reliable workspace over nightlife-heavy or amenity-focused properties.
- Boundary condition: social or leisure trips where nightlife and amenities are the goal
- Exception condition: current request explicitly asks for lively atmosphere or group entertainment

Query:

I'm booking a two-night trip to Chicago for a client workshop, with prep calls and slide edits
between sessions. Which hotel option should I choose?

Choices:

- `A`: A full-service downtown hotel with a spa, pool, and several restaurants, but a busy lobby and mixed
       reviews about room noise during events.
- `B` **[gold]**: A quiet business-oriented hotel about a 10-minute walk from the workshop, with strong Wi-Fi reviews,
                  a proper desk, and rooms away from the lobby bar.
- `C`: A lower-priced airport hotel with free breakfast and shuttle service, but a 40-minute commute to the
       workshop and basic in-room work setup.
- `D`: A trendy hotel in the nightlife district with a rooftop lounge, popular bar scene, and smaller
       rooms, about 15 minutes by rideshare from the workshop.

Evidence preview:

- `tm2_planning_v02_user_0028_s0016` | index `16` | signal `support` | source `flights`
  - user: Hi, I need help booking a round trip for a work trip next month.
  - assistant: Of course. What cities are you flying between, and what dates do you need?
  - user: I’m flying from San Diego to Boston. I need to arrive Monday, April 14, before dinner, and
    come back Thursday evening, April 17.
  - assistant: Got it. Any airline, time, or connection preferences?
  - user: I’d prefer nonstop if the price isn’t ridiculous. I have a client review Tuesday morning,
    so I don’t want to land after 7 pm on Monday. I’ll need a clear head.
- `tm2_planning_v02_user_0028_s0022` | index `22` | signal `support` | source `flights`
  - user: I need help putting together a quick work trip from Los Angeles to Chicago. Round trip,
    leaving next Monday and coming back Thursday evening if possible.
  - assistant: Sure. Which Los Angeles airport do you prefer, and are you aiming for Chicago O'Hare
    or Midway? Also, do you have preferred departure windows?
  - user: LAX is easiest. O'Hare is probably better because the meetings are downtown. I can leave
    Monday morning but not before 7, and I need to be back Thursday night.
  - assistant: I found a few reasonable options. One is LAX to ORD Monday 8:10 AM–2:15 PM nonstop,
    returning Thursday 6:40 PM–9:15 PM nonstop. Another leaves Monday 10:35 A...
  - user: Yes, nonstop only. The 8:10 Monday flight sounds better. I need time Monday afternoon to
    settle in and prepare before Tuesday meetings.
- `tm2_planning_v02_user_0028_s0032` | index `32` | signal `support` | source `flights`
  - user: Hi, I need help planning a round trip from San Francisco to Salt Lake City next month.
  - assistant: Sure. What dates are you considering, and do you prefer SFO or are nearby Bay Area
    airports acceptable?
  - user: SFO is easiest. I need to leave Monday, April 14, and come back Thursday, April 17. It is
    for a client workshop, so I cannot arrive wiped out.
  - assistant: Understood. Do you need to be in Salt Lake City by a certain time on Monday?
  - user: Yes, I have a preparation call at 2:00 pm local time. Landing late morning would be ideal.
    Carry-on only.

### 15. tm2_planning_v02_user_0001_habit_tight_schedule_nonstop_priority_p02_exception

- Probe type: `exception`
- User: `tm2_planning_v02_user_0001`
- Public probe id: `taskmaster_planning_v02_probe_29ec1155aff64789`
- Proposed gold: `D` / `apply_current_trip_constraint`
- Model decision: `revise`

Reviewer note:

> revise: The exception query and gold are coherent, but the supplied evidence preview is
> hotel-focused and does not substantiate the flight/nonstop default.

Hidden habit graph:

- Template: `tight_schedule_nonstop_priority`
- Name: tight schedules prefer nonstop flights
- Condition: travel with same-day meetings, short connection windows, or high delay sensitivity
- Default action: Prefer nonstop or lowest-transfer itineraries over slightly cheaper connecting options.
- Boundary condition: relaxed travel where transfers are acceptable and cost matters more
- Exception condition: current trip explicitly prioritizes budget or airline miles over schedule reliability

Query:

For an upcoming same-day client trip, you note that the meeting time is flexible if delays occur,
but this booking must stay under $300 out of pocket and should use your saved airline miles if
possible. Which flight should be chosen?

Choices:

- `A`: Book the $520 nonstop that arrives earliest and gives the largest schedule cushion.
- `B`: Book the $240 two-stop itinerary that arrives much later, even though it cannot use your miles.
- `C`: Book the $270 one-stop itinerary on another airline with a 35-minute connection, since it is the
       cheapest cash fare.
- `D` **[gold]**: Book the $285 one-stop itinerary on the airline where you can apply miles, with a reasonable
                  75-minute connection.

Evidence preview:

- `tm2_planning_v02_user_0001_s0001` | index `1` | signal `exception` | source `hotels`
  - user: I'd like help finding a hotel in San Diego, and I may need flights too.
  - assistant: Sure. What dates are you traveling, where in San Diego do you need to be, and do you
    have any hotel or flight constraints?
  - user: Arrive Thursday, May 16, leave Sunday, May 19. I need to be near the convention center,
    but this is not like my usual work trips where I land and go straight...
  - assistant: Got it. For hotels near the convention center, are you aiming for walking distance, a
    specific star level, or a budget range?
  - user: Walking distance would be great, but keep it under about $260 a night if possible. I don't
    need luxury, just clean, safe, and not too far from the Gaslamp area.

### 16. tm2_planning_v02_user_0007_habit_short_trip_no_checked_bag_p02_exception

- Probe type: `exception`
- User: `tm2_planning_v02_user_0007`
- Public probe id: `taskmaster_planning_v02_probe_bdf3fb6e637ce47f`
- Proposed gold: `D` / `apply_current_trip_constraint`
- Model decision: `revise`

Reviewer note:

> revise: The exception gold is correct and supported, but the distractors—especially the option
> explicitly saying to ship items even though they cannot be shipped—make the answer too obviously
> wrong rather than a plausible planning tradeoff.

Hidden habit graph:

- Template: `short_trip_no_checked_bag`
- Name: short trips avoid checked baggage
- Condition: one- or two-night trips, quick work visits, or short weekend travel
- Default action: Prefer carry-on-friendly itineraries and avoid options that require checked baggage.
- Boundary condition: longer trips or trips requiring bulky equipment
- Exception condition: current request explicitly includes items that require checked luggage

Query:

You are helping plan a one-night JFK-to-SFO work trip. The traveler says: "This time I must bring a
locked sample case and tripod that are too large for carry-on, cannot be shipped ahead, and must
arrive with me for the morning presentation." Which plan best fits these instructions?

Choices:

- `A`: Ship the sample case and tripod to the hotel, then book a carry-on-only flight, even though the
       traveler said the items cannot be shipped.
- `B`: Book the cheapest basic-economy nonstop, keep everything carry-on only, and rely on early boarding
       to avoid any bag fees.
- `C`: Book a cheaper connecting itinerary and plan to gate-check the oversized case only if the cabin bins
       are full.
- `D` **[gold]**: Book a nonstop flight that includes at least one checked bag, choose a fare with standard baggage
                  allowance, and build in time for bag drop and baggage claim.

Evidence preview:

- `tm2_planning_v02_user_0007_s0014` | index `14` | signal `exception` | source `flights`
  - user: I’m trying to clean up a few work trips at once. First is a quick Newark to Raleigh
    overnight next Monday. Please keep it simple: backpack plus roller only,...
  - assistant: For that one-night Raleigh visit, I’d focus on nonstop or one-stop flights with
    standard economy or better, since basic fares can be awkward about carry-ons....
  - user: Exactly. Same rule as my Boston client visit last month and the Chicago one before that.
    If I’m gone one night, I don’t want to stand at baggage claim.
  - assistant: Understood. For short work hops, I’ll keep the plan cabin-friendly: normal carry-on
    allowance, enough overhead-bin likelihood, and no itinerary where checkin...
  - user: Good. Second thing: Saturday morning to Sunday night in Montreal for a friend’s event.
    Flexible airports are okay, but I still want to travel light.

### 17. tm2_planning_v02_user_0026_habit_uncertain_trip_refundable_fare_p03_explicit_retrieval

- Probe type: `explicit_retrieval`
- User: `tm2_planning_v02_user_0026`
- Public probe id: `taskmaster_planning_v02_probe_26e1dadc4325b1eb`
- Proposed gold: `C` / `retrieve_supported_preference`
- Model decision: `revise`

Reviewer note:

> revise: model labeling failed; manually review (RuntimeError)

Hidden habit graph:

- Template: `uncertain_trip_refundable_fare`
- Name: uncertain trips prefer refundable fares
- Condition: work trips with tentative meetings, pending approvals, or plans that may shift
- Default action: Prefer refundable or flexible fares even when a nonrefundable fare is cheaper.
- Boundary condition: fixed personal trips with confirmed dates
- Exception condition: current request explicitly says the dates are locked and lowest fare is the priority

Query:

Which planning preference best matches the user's prior travel decisions when work travel is still
tentative or approvals/meetings may change?

Choices:

- `A`: Book upscale hotels first, then adjust flights around the hotel cancellation policy.
- `B`: Prioritize nonstop flights and short layovers over cancellation flexibility for tentative work
       trips.
- `C` **[gold]**: Select refundable or flexible flights and hotels, even when cheaper nonrefundable options are
                  available.
- `D`: Choose the lowest-priced nonrefundable flight whenever dates are known, regardless of trip type.

Evidence preview:

- `tm2_planning_v02_user_0026_s0007` | index `7` | signal `support` | source `hotels`
  - user: I need help planning a possible work trip to Palm Springs. It would be from Seattle,
    probably the week of April 14, but the client still has to confirm the m...
  - assistant: Sure. Since the dates are not fully confirmed, do you want me to look at both hotel
    and flight options with flexibility in mind?
  - user: Yes, please. I’ll need a hotel near downtown Palm Springs and a flight into either Palm
    Springs or Ontario if the timing is better.
  - assistant: For the hotel, are you aiming for a particular budget or style?
  - user: Something comfortable for work, ideally under $260 a night, with reliable Wi-Fi and
    parking. I don’t need a resort experience.
- `tm2_planning_v02_user_0026_s0019` | index `19` | signal `support` | source `hotels`
  - user: I may need to be in New Orleans for a client meeting next month. Can you help me look at a
    hotel for Friday through Monday morning for two people?
  - assistant: Sure. What area would you prefer, and is this a work trip or a personal stay?
  - user: Work trip, but my spouse may come along if the schedule holds. The meeting is supposed to
    be near the Central Business District, possibly close to Poydras St...
  - assistant: Got it. For two guests, Friday check-in and Monday checkout near the CBD. Do you have
    a nightly budget or hotel style in mind?
  - user: Ideally under $260 a night before taxes, but I care more about being able to change or
    cancel. The client has already moved this meeting once.
- `tm2_planning_v02_user_0026_s0024` | index `24` | signal `support` | source `flights`
  - user: Hi, can you help me look at Delta flights to Seattle? This would be for a client meeting,
    but the schedule is still a little unsettled.
  - assistant: Of course. What city are you flying from, and what dates should I check first?
  - user: From Atlanta to Seattle. The team is talking about going out Monday, March 18, and
    returning Thursday, March 21, but my manager has not fully approved the on...
  - assistant: Got it. I’ll look at ATL to SEA on Delta for March 18 to March 21. Do you prefer
    nonstop flights, certain times of day, or a particular cabin?
  - user: Nonstop is preferred if it’s reasonable. Main Cabin is fine. I don’t want Basic Economy
    for this because the agenda could slide by a day.

### 18. tm2_planning_v02_user_0027_habit_work_hotel_near_venue_p03_explicit_retrieval

- Probe type: `explicit_retrieval`
- User: `tm2_planning_v02_user_0027`
- Public probe id: `taskmaster_planning_v02_probe_c04686d4c37d310c`
- Proposed gold: `C` / `retrieve_supported_preference`
- Model decision: `revise`

Reviewer note:

> revise: model labeling failed; manually review (RuntimeError)

Hidden habit graph:

- Template: `work_hotel_near_venue`
- Name: work hotels prefer venue proximity
- Condition: work, conference, client, or onsite meeting hotel selection
- Default action: Prefer hotels close to the meeting venue over cheaper hotels with a longer commute.
- Boundary condition: leisure trips where neighborhood character or sightseeing access matters more
- Exception condition: current request explicitly prioritizes budget or a specific neighborhood over commute time

Query:

Based on the user's prior travel planning, which hotel choice should be favored when arranging
lodging for a work conference, client visit, or onsite meeting?

Choices:

- `A`: Choose the lowest nightly rate available, even if it means staying farther from the meeting venue.
- `B`: Choose a hotel near the airport to make arrival and departure easier, even if the meeting venue is
       downtown.
- `C` **[gold]**: Choose a hotel within walking distance or a few blocks of the meeting venue, even if cheaper options
                  would require a longer commute.
- `D`: Choose a hotel in the most distinctive neighborhood for restaurants and sightseeing, even when the
       trip is work-focused.

Evidence preview:

- `tm2_planning_v02_user_0027_s0002` | index `2` | signal `support` | source `flights`
  - user: I need help arranging a quick work trip to Dallas next month. Cheapest flights are fine,
    but I need onboard Wi-Fi or at least a usable hotspot connection bec...
  - assistant: Sure. What are your travel dates, departure airport, and any timing constraints?
  - user: One traveler, SFO to DFW. Leave Tuesday May 14, come back Thursday May 16. I need to be at
    a client workshop by 10:30 a.m. Wednesday, so arriving Tuesday eve...
  - assistant: I found a few reasonable nonstop options. The lowest fare is a late Tuesday flight
    arriving 11:20 p.m. with Wi-Fi listed. There is also a 4:10 p.m. departure...
  - user: Let's take the 4:10 p.m. outbound. I don't want to land close to midnight before a client
    day. The 7:15 p.m. return is fine.
- `tm2_planning_v02_user_0027_s0003` | index `3` | signal `support` | source `hotels`
  - user: I need help booking a hotel in San Francisco for a work trip next month.
  - assistant: Sure. What dates are you traveling, and where will your meetings be?
  - user: Checking in Tuesday, May 14 and checking out Friday, May 17. The meetings are at Moscone
    West, and I'll have 8:00 a.m. sessions each day.
  - assistant: Got it. Do you have a target nightly rate, preferred hotel class, or loyalty program?
  - user: Company pays, but I should keep it reasonable. I'd rather not save thirty dollars if it
    means a long ride before the first session.
- `tm2_planning_v02_user_0027_s0017` | index `17` | signal `support` | source `hotels`
  - user: I need help picking a hotel in Chicago for a short work trip next month.
  - assistant: Sure. What dates are you traveling, and where will you need to be for work?
  - user: Two nights, September 9th to 11th. The client meetings are at 500 West Madison in the West
    Loop, starting at 8:00 each morning.
  - assistant: Got it. Is there a nightly budget, and do you care more about amenities,
    neighborhood, or commute time?
  - user: It is reimbursed, so I don't need the cheapest place. I want something clean and quiet,
    but the main thing is being close to the client office. I hate adding...

### 19. tm2_planning_v02_user_0028_habit_quiet_hotel_for_work_p03_explicit_retrieval

- Probe type: `explicit_retrieval`
- User: `tm2_planning_v02_user_0028`
- Public probe id: `taskmaster_planning_v02_probe_2976d535c131f020`
- Proposed gold: `C` / `retrieve_supported_preference`
- Model decision: `revise`

Reviewer note:

> revise: The gold matches the hidden default, but the shown support evidence is flight-planning/prep
> context rather than lodging choices demonstrating a quiet work-friendly hotel preference.

Hidden habit graph:

- Template: `quiet_hotel_for_work`
- Name: work hotels prefer quiet properties
- Condition: business travel or trips requiring preparation, calls, or focused work
- Default action: Prefer quieter hotels with reliable workspace over nightlife-heavy or amenity-focused properties.
- Boundary condition: social or leisure trips where nightlife and amenities are the goal
- Exception condition: current request explicitly asks for lively atmosphere or group entertainment

Query:

When the user's trip involves business travel, prep calls, or focused work, which lodging option
should be prioritized based on their previous choices?

Choices:

- `A`: A resort-style property with the strongest pool, spa, and social amenities, even if the room setup
       is less work-friendly.
- `B`: A centrally located hotel near nightlife, restaurants, and bars, even if it is louder at night.
- `C` **[gold]**: A quieter hotel with reliable Wi-Fi and a real workspace, even if it has fewer nightlife or
                  entertainment features.
- `D`: The lowest-cost hotel available, regardless of noise level, desk setup, or internet reliability.

Evidence preview:

- `tm2_planning_v02_user_0028_s0016` | index `16` | signal `support` | source `flights`
  - user: Hi, I need help booking a round trip for a work trip next month.
  - assistant: Of course. What cities are you flying between, and what dates do you need?
  - user: I’m flying from San Diego to Boston. I need to arrive Monday, April 14, before dinner, and
    come back Thursday evening, April 17.
  - assistant: Got it. Any airline, time, or connection preferences?
  - user: I’d prefer nonstop if the price isn’t ridiculous. I have a client review Tuesday morning,
    so I don’t want to land after 7 pm on Monday. I’ll need a clear head.
- `tm2_planning_v02_user_0028_s0022` | index `22` | signal `support` | source `flights`
  - user: I need help putting together a quick work trip from Los Angeles to Chicago. Round trip,
    leaving next Monday and coming back Thursday evening if possible.
  - assistant: Sure. Which Los Angeles airport do you prefer, and are you aiming for Chicago O'Hare
    or Midway? Also, do you have preferred departure windows?
  - user: LAX is easiest. O'Hare is probably better because the meetings are downtown. I can leave
    Monday morning but not before 7, and I need to be back Thursday night.
  - assistant: I found a few reasonable options. One is LAX to ORD Monday 8:10 AM–2:15 PM nonstop,
    returning Thursday 6:40 PM–9:15 PM nonstop. Another leaves Monday 10:35 A...
  - user: Yes, nonstop only. The 8:10 Monday flight sounds better. I need time Monday afternoon to
    settle in and prepare before Tuesday meetings.
- `tm2_planning_v02_user_0028_s0032` | index `32` | signal `support` | source `flights`
  - user: Hi, I need help planning a round trip from San Francisco to Salt Lake City next month.
  - assistant: Sure. What dates are you considering, and do you prefer SFO or are nearby Bay Area
    airports acceptable?
  - user: SFO is easiest. I need to leave Monday, April 14, and come back Thursday, April 17. It is
    for a client workshop, so I cannot arrive wiped out.
  - assistant: Understood. Do you need to be in Salt Lake City by a certain time on Monday?
  - user: Yes, I have a preparation call at 2:00 pm local time. Landing late morning would be ideal.
    Carry-on only.
