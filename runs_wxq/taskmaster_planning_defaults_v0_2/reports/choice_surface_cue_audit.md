# Choice Surface Cue Audit

Audit target: final merged `public/probes.jsonl`.

- Probes: 120
- Choices: 480
- Gold choices with contrast markers: 14/120 (11.7%)
- Distractor choices with contrast markers: 74/360 (20.6%)
- Gold choices with broad negative/caveat cues: 28/120 (23.3%)
- Distractor choices with broad negative/caveat cues: 104/360 (28.9%)
- Probes where >=2 distractors have contrast markers and gold has none: 17
- Probes where >=2 distractors have broad negative/caveat cues and gold has none: 19

## Marker Counts

Gold marker types:
- `but`: 6
- `though`: 4
- `rather_than`: 4
- `even_though`: 2

Distractor marker types:
- `but`: 41
- `though`: 24
- `even_though`: 22
- `while`: 4
- `despite`: 4
- `rather_than`: 1

## Probe Type Breakdown

| Probe type | Role | Total | Contrast markers | Marker rate | Broad cue rate |
|---|---|---:|---:|---:|---:|
| `boundary` | distractor | 90 | 34 | 37.8% | 44.4% |
| `boundary` | gold | 30 | 3 | 10.0% | 23.3% |
| `direct_use` | distractor | 90 | 23 | 25.6% | 30.0% |
| `direct_use` | gold | 30 | 5 | 16.7% | 20.0% |
| `exception` | distractor | 90 | 16 | 17.8% | 23.3% |
| `exception` | gold | 30 | 2 | 6.7% | 10.0% |
| `explicit_retrieval` | distractor | 90 | 1 | 1.1% | 17.8% |
| `explicit_retrieval` | gold | 30 | 4 | 13.3% | 40.0% |

## Highest-Risk Probes

### tm2_planning_v02_user_0003_habit_work_hotel_near_venue_p00_direct_use (direct_use)

- `A` distractor, markers: but: A higher-end hotel near Lady Bird Lake, $260/night, with a pool and nicer rooms but a 25-minute walk or short rideshare to the venue.
- `B` distractor, markers: but: A boutique hotel in East Austin, $225/night, with better restaurants nearby but a 20-minute rideshare to the convention center.
- `C` gold, markers: none: A downtown hotel about a 5-minute walk from the convention center, $285/night, with standard rooms and basic amenities.
- `D` distractor, markers: but: An airport-area hotel, $175/night, with free breakfast and a shuttle, but a 35–45 minute commute each way.

### tm2_planning_v02_user_0003_habit_work_hotel_near_venue_p02_exception (exception)

- `A` distractor, markers: none: A Cambridge boutique hotel with highly rated amenities, about 35 minutes from the convention center and not in the requested neighborhoods.
- `B` distractor, markers: but: An airport-area hotel with the lowest nightly rate, but a 45-minute commute and no easy access to Back Bay or Fenway.
- `C` gold, markers: none: A Back Bay hotel near Copley Square, about 25 minutes to the convention center by transit, with rates in the middle of the available options.
- `D` distractor, markers: but: A Seaport hotel two blocks from the convention center, but outside Back Bay/Fenway and significantly more expensive.

### tm2_planning_v02_user_0004_habit_quiet_hotel_for_work_p00_direct_use (direct_use)

- `A` distractor, markers: none: A hotel attached to the workshop venue, with the shortest commute, a popular lobby bar, compact rooms, and desk space that reviewers say is better for quick emails than longer work blocks.
- `B` distractor, markers: though: A newer hotel in a restaurant-heavy area with a rooftop lounge, strong design reviews, and easy access to dinner spots, though guests often mention a lively evening atmosphere.
- `C` distractor, markers: but: A full-service property a short rideshare from the workshop, with a gym, pool, breakfast package, and business center, but mixed room reviews for Wi-Fi strength and work surfaces.
- `D` gold, markers: none: A smaller business-oriented hotel on an office-heavy block, about a 10-minute walk from the workshop, with reliable Wi-Fi, full in-room desks, and fewer on-site amenities or nightlife options.

### tm2_planning_v02_user_0007_habit_short_trip_no_checked_bag_p00_direct_use (direct_use)

- `A` distractor, markers: but: Take the late nonstop Tuesday morning to save on hotel cost Monday night, but arrive only 70 minutes before the client meeting and use a farther airport-area hotel.
- `B` distractor, markers: but: Book the cheapest basic-economy fare with a connection each way, saving about $140, but with last-group boarding and only a personal item clearly included.
- `C` distractor, markers: but: Use the lower-priced regional-jet connection that arrives 50 minutes earlier, but the aircraft commonly requires roller bags to be gate-checked.
- `D` gold, markers: none: Take the nonstop main-cabin flights both ways, pay about $85 more, and choose an aisle seat with standard overhead-bin access; stay at the hotel two blocks from the client office.

### tm2_planning_v02_user_0008_habit_international_long_layover_buffer_p01_boundary (boundary)

- `A` distractor, markers: though, even_though: Recommend the Charlotte option because the 3-hour layover provides the largest buffer, even though it arrives later.
- `B` gold, markers: none: Recommend the nonstop flight because it is the simplest and fastest domestic option with the lowest connection risk.
- `C` distractor, markers: while: Recommend the Atlanta option because it saves $19 compared with the nonstop while still arriving reasonably early.
- `D` distractor, markers: though, even_though: Recommend the Chicago option because it is the cheapest, even though it adds a long detour and arrives much later.

### tm2_planning_v02_user_0013_habit_tight_schedule_nonstop_priority_p01_boundary (boundary)

- `A` distractor, markers: despite: Select a fully refundable fare on the nonstop flight to maximize flexibility despite the higher price.
- `B` gold, markers: none: Book the comfortable one-stop itinerary because the schedule is flexible and it offers better overall value.
- `C` distractor, markers: none: Choose a flight to a farther alternate airport to get the absolute lowest fare, even if it adds a long drive.
- `D` distractor, markers: though, even_though: Book the nonstop flight because it reduces transfer risk, even though it costs substantially more.

### tm2_planning_v02_user_0013_habit_tight_schedule_nonstop_priority_p02_exception (exception)

- `A` distractor, markers: but: A nonstop flight into an alternate airport that saves a little cash but requires a long rental-car drive before the meeting.
- `B` gold, markers: none: A one-stop itinerary on my preferred airline using miles plus minimal taxes, with a 95-minute connection and arrival about 2.5 hours before the meeting.
- `C` distractor, markers: none: The absolute cheapest cash fare with two connections, including a 35-minute layover, arriving just under an hour before the meeting.
- `D` distractor, markers: but: A nonstop flight arriving 4 hours before the meeting, but requiring a substantially higher cash fare and not using any miles.

### tm2_planning_v02_user_0014_habit_uncertain_trip_refundable_fare_p00_direct_use (direct_use)

- `A` gold, markers: none: Book the refundable economy fare for $620 with no change fees and a full refund to the original payment method.
- `B` distractor, markers: though, even_though: Book the basic economy fare for $390 because it is the cheapest, even though it is nonrefundable and cannot be changed.
- `C` distractor, markers: but: Book the standard economy fare for $455 because it includes a carry-on, but changes only receive airline credit and refunds are not allowed.
- `D` distractor, markers: none: Wait until the approval is finalized before booking, even if the fare may rise and the preferred flight times could sell out.

### tm2_planning_v02_user_0014_habit_uncertain_trip_refundable_fare_p01_boundary (boundary)

- `A` distractor, markers: though, even_though: Choose a premium economy fare because the longer flight will be more comfortable, even though the traveler prioritized price.
- `B` distractor, markers: despite: Delay booking until closer to the wedding in case fares drop, despite the fixed dates and confirmed plans.
- `C` gold, markers: none: Choose the standard nonrefundable economy fare on the confirmed dates because it is the lowest reasonable price and the schedule is fixed.
- `D` distractor, markers: though, even_though: Choose a fully refundable economy fare even though it costs more, to protect against possible date changes.

### tm2_planning_v02_user_0016_habit_quiet_hotel_for_work_p00_direct_use (direct_use)

- `A` distractor, markers: but: A stylish downtown hotel attached to a popular rooftop bar, close to restaurants and nightlife, but with smaller rooms and more evening noise.
- `B` gold, markers: none: A business-oriented hotel a few blocks from the main dining area, with confirmed strong Wi-Fi, a proper in-room desk, and rooms available away from elevators.
- `C` distractor, markers: but: A lower-cost airport hotel with free breakfast and shuttle service, but mixed reviews on Wi-Fi reliability and a less convenient location for meetings.
- `D` distractor, markers: but: A larger resort-style property with a pool, spa, and lively lobby scene, but a longer commute to the client site and limited desk space in standard rooms.

### tm2_planning_v02_user_0018_habit_family_trip_flexible_cancellation_p00_direct_use (direct_use)

- `A` distractor, markers: though, even_though: Book only the flights now and wait to reserve lodging in case hotel prices drop, even though family-size rooms may become limited.
- `B` distractor, markers: none: The lowest-priced package with basic economy flights and a prepaid, nonrefundable hotel, saving about $420 compared with the next option.
- `C` distractor, markers: but: A premium package with extra-legroom seats and a fully refundable resort stay, but it costs about $1,200 more than the cheapest package.
- `D` gold, markers: none: A package costing about $300 more with standard economy flights that can be changed for fare differences and a hotel cancellable until three days before arrival.

### tm2_planning_v02_user_0021_habit_red_eye_avoidance_p01_boundary (boundary)

- `A` distractor, markers: though, even_though: Recommend the Friday 6:00 a.m. one-stop because it is the lowest fare, even though it arrives after the requested time.
- `B` gold, markers: none: Recommend the Thursday 10:45 p.m. red-eye because it preserves Thursday daytime and arrives before 8:30 a.m. Friday.
- `C` distractor, markers: none: Recommend the Thursday 1:30 p.m. nonstop because it avoids overnight travel and is slightly cheaper.
- `D` distractor, markers: though, even_though: Recommend the Wednesday 11:30 p.m. red-eye because it avoids traveling during Thursday daytime, even though it arrives a day early and may require an extra hotel night.

### tm2_planning_v02_user_0024_habit_business_travel_arrival_buffer_p01_boundary (boundary)

- `A` distractor, markers: despite: Choose a red-eye arriving very early Friday morning to maximize unused time in the city, despite likely making the first day tiring.
- `B` gold, markers: none: Choose the cheaper nonstop Friday evening flight that arrives around 9:30 p.m., paired with a reasonably priced hotel near the subway line they will use on Saturday.
- `C` distractor, markers: though, even_though: Choose a flight with a long layover that arrives mid-afternoon Friday because it creates an extra arrival cushion, even though it is less convenient.
- `D` distractor, markers: though, even_though: Choose a much more expensive early-morning Friday flight so they arrive before noon, even though they have no plans until the next day.

### tm2_planning_v02_user_0025_habit_tight_schedule_nonstop_priority_p01_boundary (boundary)

- `A` gold, markers: none: A one-stop itinerary with a 2-hour layover that saves $165 and arrives in the evening.
- `B` distractor, markers: none: A nonstop flight on the traveler’s preferred airline that costs $220 more and earns extra miles.
- `C` distractor, markers: but: A nonstop flight that costs $175 more but has the lowest delay exposure.
- `D` distractor, markers: but: A two-stop itinerary with 40-minute connections that saves $190 but has little room for delays.

### tm2_planning_v02_user_0028_habit_quiet_hotel_for_work_p00_direct_use (direct_use)

- `A` distractor, markers: but: The official workshop hotel downtown, with the shortest commute, a gym, and several restaurants, but an active lobby bar and multiple reviews mentioning noise during events.
- `B` gold, markers: none: A smaller business-oriented hotel about a 10-minute walk from the workshop, with strong Wi-Fi reviews, a proper desk, and quieter rooms away from the bar and elevators.
- `C` distractor, markers: but: A lower-priced airport hotel with free breakfast and shuttle service, but a 40-minute commute to the workshop and only a basic in-room work setup.
- `D` distractor, markers: none: A stylish hotel in a nightlife-heavy area with a rooftop lounge and popular restaurant, compact rooms, and a roughly 15-minute rideshare to the workshop.

### tm2_planning_v02_user_0028_habit_quiet_hotel_for_work_p01_boundary (boundary)

- `A` gold, markers: none: A South Beach hotel with a lively pool, rooftop bar, suite-style or adjoining room options, and easy walks to clubs and late-night restaurants.
- `B` distractor, markers: none: A quiet hotel in the financial district with large in-room desks, fast Wi-Fi, and subdued common areas, about 20 minutes by rideshare from South Beach.
- `C` distractor, markers: but: A mid-priced airport hotel with free breakfast and a shuttle, but limited nearby restaurants and little late-night activity.
- `D` distractor, markers: but: A beachfront resort in a quieter northern area with a spa and polished rooms, but few nightlife options within walking distance.

### tm2_planning_v02_user_0029_habit_leisure_relaxed_pacing_p01_boundary (boundary)

- `A` gold, markers: none: Use a work-centered plan: take a Sunday midday nonstop that lands by midafternoon, book a hotel near the workshop area, and schedule the return after the Tuesday debrief with normal airport buffer.
- `B` distractor, markers: though, even_though: Arrive Saturday late afternoon and keep Sunday mostly open to settle in slowly before the prep dinner, even though it adds an extra hotel night before the work commitments begin.
- `C` distractor, markers: none: Take a shorter-trip option that lands around 5:10 PM Sunday and stay near the dinner venue, accepting a tighter margin before the 7:00 PM prep dinner.
- `D` distractor, markers: but: Take a Sunday midday arrival but choose a downtown or waterfront hotel for better evening options, then use rideshares to reach the workshop sessions in Mission Valley.

