#pragma once

// Rules that turn what the game reports (frame time, moving bodies, impacts) into the wind field's inputs. No game
// dependency: WindField.cpp calls them with game values, the offline check (tools/offline/WindSimCheck.cpp) with
// values set by hand, so the check runs the plugin's own decisions.

#include "WindShaders.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace WindRules
{
	inline constexpr float kSqrtPi = 1.7724539f;
	inline constexpr float kTwoPi = 6.2831853f;

	// ---- time

	// Simulated time follows real time down to 15 frames a second; below that the field runs slow rather than
	// taking steps it was not checked at.
	inline constexpr float kLongestFrame = 1.0f / 15.0f;
	// A frame longer than this is taken in two steps of half its length. Measured against the same 2 seconds at 60
	// frames a second (check "2 s at ... fps match 60 fps", tolerance 0.02 on the fog amount in a runner's lane,
	// with FogRefill 0.06 and ActorFollow 6): one step of 1/30 s 0.018, of 1/28 s 0.022, of 1/20 s 0.035, of 1/15 s
	// 0.046; two steps at 20 and 15 frames a second 0.011 and 0.020. So 1/30 is the longest single step, and a game
	// held at 30 frames a second pays the second step on the frames that run a little long.
	inline constexpr float kLongestStep = 1.0f / 30.0f;

	inline int StepsPerFrame(float a_frameSeconds)
	{
		return a_frameSeconds > kLongestStep ? 2 : 1;
	}

	// ---- the layer's reach

	// A body or an impact disturbs the layer only from inside it. The layer's top is at most Thickness above the
	// ground; twice that is taken as its reach: the height grid has one sample every 64 units and reads a ledge, a
	// stair or a river bank up to about a cell too low or too high, and a jump (about 1 m) must not switch a
	// runner's trail off. Thickness 126 gives 252 units (3.6 m): a dragon in the air, someone on a bridge or a wall
	// walk and a spell hitting high on a cliff are out. Where the ground is not known the answer is yes, as before.
	// Pooled fog and steam (the tall layer) can stand higher than this; a body above the reach leaves them alone.
	// Never less than kLeastReach: with a thin layer set in the ini a jump (about 76 units) plus one height cell (64)
	// would switch a runner's trail off (review 2026-11-14).
	inline constexpr float kLeastReach = 150.0f;
	inline float LayerReach(float a_thickness)
	{
		return std::max(2.0f * a_thickness, kLeastReach);
	}

	// A heat source that clears the fog (WindField.cpp FillBlasts: a fire, a fire spell's burning ground, a held torch).
	struct Heat
	{
		float x, y, z, radius;
		float rate;   // the blast rate it clears at (FireClear or TorchClear)
	};

	// The heat sources that get a blast slot, at most a_slots, from a list sorted nearest first. One slot per spot: a
	// source whose centre lies within half the larger reach of one already kept merges into it (the pieces of a flame
	// wall, a hazard laid over a campfire, a torch by the fire; Carbon review 2026-11-17: they took every slot and
	// doubled a campfire's rate), and the spot keeps the stronger of the two, the higher rate and then the larger reach
	// (Codex review 2026-11-17: a torch nearer the centre than its campfire had replaced the fire). A source that finds
	// no slot and no spot is left out. O(n x slots).
	inline std::vector<Heat> PickHeat(const std::vector<Heat>& a_nearestFirst, std::size_t a_slots)
	{
		std::vector<Heat> kept;
		for (const auto& h : a_nearestFirst) {
			const auto same = std::find_if(kept.begin(), kept.end(),
				[&](const Heat& a_k) { return std::hypot(a_k.x - h.x, a_k.y - h.y) < 0.5f * std::max(a_k.radius, h.radius); });
			if (same != kept.end()) {
				if (h.rate > same->rate || (h.rate == same->rate && h.radius > same->radius)) *same = h;
				continue;
			}
			if (kept.size() < a_slots) kept.push_back(h);
		}
		return kept;
	}

	inline bool WithinLayer(float a_z, bool a_groundKnown, float a_ground, float a_thickness)
	{
		return !a_groundKnown || !(a_z - a_ground > LayerReach(a_thickness));
	}

	// ---- moving bodies

	// The lane ActorRadius 70 opens is sqrt(2) x 70 = 99 units wide (1.4 m): a person and a horse fit in it and stay
	// as they were. A body wider than that gets a radius that makes the lane as wide as the body, and a deeper lane.
	// Both stop at 4 lanes (396 units, 5.7 m; radius 280): the width comes from the game's bounds, which were not
	// seen in game yet (a dragon's may span its wings), and the fog at a body thins further the wider its lane is.
	inline constexpr float kPersonLane = 99.0f;
	inline constexpr float kWidestBody = 4.0f;   // lanes up to 4 x a person's (a mammoth, a dragon on the ground)
	inline constexpr float kStillSpeed = 5.0f;   // units/s; slower than this is standing still: no drag

	struct Body
	{
		float x, y, z;        // feet, world units
		float vx, vy;         // units/s
		float width;          // the smaller horizontal side of its bounds, units (0 = not known: a person)
		bool  rider;          // sits on a mount: the mount stirs the fog for both
		bool  groundKnown;
		float ground;         // the field's ground height at (x, y)
	};

	// The smaller horizontal side of a body's bounds: its width. The larger one is the length of anything on four legs.
	inline float BodyWidth(float a_minX, float a_minY, float a_maxX, float a_maxY)
	{
		const float w = std::min(a_maxX - a_minX, a_maxY - a_minY);
		return std::isfinite(w) && w > 0.0f ? w : 0.0f;
	}

	inline float DragRadius(float a_actorRadius, float a_bodyWidth)
	{
		return std::max(a_actorRadius, std::min(a_bodyWidth, kWidestBody * kPersonLane) / 1.4142136f);
	}

	inline float BodyScale(float a_bodyWidth)
	{
		return std::clamp(a_bodyWidth / kPersonLane, 1.0f, kWidestBody);
	}

	// How much more fog a wide body removes from its path than a person. 1: with sqrt(BodyScale) a body 4 lanes wide
	// thinned the fog round itself to 0.12 (a person: 0.46), which reads as the fog opening wide for it; real fog does
	// not (user 2026-11-13). The game's bounds were not seen in game yet either (review 2026-11-14).
	inline float DepthScale(float /*a_bodyWidth*/)
	{
		return 1.0f;
	}

	// HLSL smoothstep: 0 below a_lo, 1 above a_hi, 3t^2 - 2t^3 between.
	inline float SmoothStep(float a_lo, float a_hi, float a_x)
	{
		const float t = std::clamp((a_x - a_lo) / (a_hi - a_lo), 0.0f, 1.0f);
		return t * t * (3.0f - 2.0f * t);
	}

	// The share of the push a body moving at a_speed gets, the same formula as the shader's (WindShaders.h Source):
	// a_walkShare (DragParams.x) fades in from 0.18 to 0.36 x a_runSpeed (DragParams.w; the game's walk, 80.1 units/s,
	// is just past it at 220), the rest from a_runSpeed to 1.5 x it. For the profiler log; WindSimCheck.cpp
	// TestDragGate reads the shader's own value and compares.
	inline float DragGate(float a_speed, float a_walkShare, float a_runSpeed)
	{
		const float run = std::max(a_runSpeed, 1.0f);
		return a_walkShare * SmoothStep(0.18f * run, 0.36f * run, a_speed) + (1.0f - a_walkShare) * SmoothStep(run, 1.5f * run, a_speed);
	}

	inline bool Disturbs(const Body& a_body, float a_thickness)
	{
		return !a_body.rider && a_body.vx * a_body.vx + a_body.vy * a_body.vy >= kStillSpeed * kStillSpeed &&
		       WithinLayer(a_body.z, a_body.groundKnown, a_body.ground, a_thickness);
	}

	// One drag row. a_actorPush is the fog amount removed on a person's path (with the air standing still the fog
	// there would thin to exp(-a_actorPush) when running); a body s = BodyScale lanes wide removes
	// a_actorPush x sqrt(s): 4 lanes wide, twice the amount. x s would ask for banks of exp(1.4 x 0.446) = 1.87, over
	// the cap of 2 wherever the layer's own banks are above 1.07. The shader's source is z x speed x Ricker(R), whose
	// integral along the path is z x sqrt(pi) x R / 2 (WindShaders.h Ricker), hence the division.
	// Measured at the game's texel (WindSimCheck.cpp TestActorPush, 370 units/s): the air a body drags along carries
	// the thinned fog with it, so the fog is thinnest just behind the body: 0.46 for a person (banks 1.12), 0.12 for a
	// body 4 lanes wide (banks 1.44; 0.30 and 1.21 with the plain push).
	// a_step, a_run: the distance the body really moved this frame and its straight run so far, this frame included (units;
	// WindShaders.h WakeDose). Read only while the wake is on (dragParams.z > 0); 0 = no wake from this body.
	inline void SetDrag(WindShaders::SimCB& a_cb, int a_slot, const Body& a_body, float a_actorRadius, float a_actorPush, float a_step = 0.0f,
		float a_run = 0.0f)
	{
		const float radius = std::max(DragRadius(a_actorRadius, a_body.width), 1.0f);
		a_cb.drags[a_slot][0] = a_body.x;
		a_cb.drags[a_slot][1] = a_body.y;
		a_cb.drags[a_slot][2] = a_body.vx;
		a_cb.drags[a_slot][3] = a_body.vy;
		a_cb.dragSize[a_slot][0] = 1.0f / (radius * radius);
		// The amount removed is a person's whatever the width (DepthScale): a wide body's lane is wider, not deeper.
		a_cb.dragSize[a_slot][1] = a_actorPush * DepthScale(a_body.width) / (0.5f * kSqrtPi * radius);
		a_cb.dragSize[a_slot][2] = std::max(a_step, 0.0f);
		a_cb.dragSize[a_slot][3] = std::max(a_run, 0.0f);
	}

	// ---- wake evaporation (WindShaders.h WakeDose, the same formula; DEBUG-LEDGER.md 2026-11-14 15:xx)

	// A heading change over this from one frame to the next starts a new straight run (WindField.cpp FillActors).
	inline constexpr float kWakeTurnCos = 0.8660254f;   // cos 30 deg
	// The wake alone never takes rho under this x rho_eq (the shader's kWakeFloor).
	inline constexpr float kWakeFloor = 0.15f;

	// The dose drag row a_slot gives the point (a_x, a_y) this frame (before x WakeEvaporate / steps), with the air there
	// moving at (a_vx, a_vy) (cb.shift[2] = the frame's field time). Straight run: the doses a parcel of air collects add
	// up to at most gate x exp(-c^2 / R^2).
	inline float WakeDose(const WindShaders::SimCB& a_cb, int a_slot, float a_x, float a_y, float a_vx = 0.0f, float a_vy = 0.0f)
	{
		const float* drag = a_cb.drags[a_slot];
		const float* size = a_cb.dragSize[a_slot];
		const float  step = size[2];
		const float  speed = std::hypot(drag[2], drag[3]);
		if (!(step > 0.0f) || speed < 1.0f) {
			return 0.0f;
		}
		const float dx = a_x - drag[0], dy = a_y - drag[1];
		const float b = -(dx * drag[2] + dy * drag[3]) / speed;
		if (b > size[3]) {
			return 0.0f;
		}
		const float R = 1.0f / std::sqrt(size[0]);
		const float c2 = std::max(dx * dx + dy * dy - b * b, 0.0f);
		const float gate = DragGate(speed, a_cb.dragParams[0], a_cb.dragParams[3]);
		const float rel = std::max(step - (a_vx * drag[2] + a_vy * drag[3]) / speed * a_cb.shift[2], 0.0f);
		return gate * std::exp(-c2 * size[0]) * (SmoothStep(0.5f * R, 2.0f * R, b) - SmoothStep(0.5f * R, 2.0f * R, b - rel));
	}

	// ---- impacts

	// An impact acts for a_duration seconds with a Hann-window pulse in time,
	//     w(t) = (1 - cos(2 pi t / T)) / T   for 0 <= t <= T,   integral over the pulse = 1:
	// it starts and ends at rest and peaks at T / 2. Rates are (setting) x w(t), so the settings are totals:
	// BlastDepth is the fog amount removed at the centre over the pulse (the fog there thins to about
	// exp(-BlastDepth) before it refills), BlastPush the speed the air gains along the hit direction.
	inline float Pulse(float a_age, float a_duration)
	{
		if (a_age < 0.0f || a_age > a_duration || a_duration <= 0.0f) {
			return 0.0f;
		}
		return (1.0f - std::cos(kTwoPi * a_age / a_duration)) / a_duration;
	}

	// A sustained hit (flame, beam) is held at the pulse's peak and eases from there down to kSustain of it over
	// kSustainEase seconds: held at the full peak it would blow at about BlastPush x 2 / T / (WindRelax +
	// VelocityDissipation), some 10 m/s with the defaults (reviewer's estimate, not measured).
	// a_heldSeconds: how long it has been held (negative = not held).
	inline constexpr float kSustain = 0.3f;
	inline constexpr float kSustainEase = 0.5f;

	inline float Sustained(float a_heldSeconds)
	{
		const float held = a_heldSeconds >= 0.0f ? std::clamp(a_heldSeconds / kSustainEase, 0.0f, 1.0f) : 0.0f;
		return 1.0f - (1.0f - kSustain) * held;
	}

	inline constexpr float kShoutKick = 1.5f;   // a shout pushes the air 1.5 x as hard along its direction; its hole is a hit's

	struct Blast
	{
		float x, y;
		float dirX, dirY;   // the hit's direction (any length; 0 = none)
		float radius;
		bool  fire;         // boils fog off instead of piling it on the rim
		bool  shout;
	};

	// One impact row at the pulse rate a_rate (1/s, Pulse x Sustained). The depth a_depth is the whole fog amount
	// removed at the centre. A fire hit takes a_fireEvaporate of it (at most all) as evaporation, a sink with the
	// impact's Gaussian footprint that asks for no air movement, and only the rest as the outward push: with 4 and 3
	// the centre thins as before, the rim rises to 1.01 instead of 1.63 and 5 % of the fog on the check's grid is gone
	// rather than moved (WindSimCheck.cpp TestImpactKinds).
	inline void SetBlast(WindShaders::SimCB& a_cb, int a_slot, const Blast& a_blast, float a_rate, float a_depth, float a_push, float a_fireEvaporate)
	{
		const float evaporate = a_blast.fire ? std::clamp(a_fireEvaporate, 0.0f, std::max(a_depth, 0.0f)) : 0.0f;
		const float len = std::hypot(a_blast.dirX, a_blast.dirY);
		// A direction that is not finite gives no kick (NaN x 0 would put NaN into the whole field; review 2026-11-14).
		const bool  directed = std::isfinite(len) && len > 1e-3f;
		const float kick = directed ? a_push * a_rate * (a_blast.shout ? kShoutKick : 1.0f) / len : 0.0f;
		const float radius = std::max(a_blast.radius, 1.0f);
		a_cb.blasts[a_slot][0] = a_blast.x;
		a_cb.blasts[a_slot][1] = a_blast.y;
		a_cb.blasts[a_slot][2] = 1.0f / (radius * radius);
		a_cb.blasts[a_slot][3] = (a_depth - evaporate) * a_rate;
		a_cb.blastKicks[a_slot][0] = directed ? a_blast.dirX * kick : 0.0f;
		a_cb.blastKicks[a_slot][1] = directed ? a_blast.dirY * kick : 0.0f;
		a_cb.blastKicks[a_slot][2] = evaporate * a_rate;
		a_cb.blastKicks[a_slot][3] = 0.0f;
	}

	// The radius of an explosion's impact from the game's explosion radius. Measured (WindSimCheck.cpp
	// TestImpactKinds): with a scale of 0.8 the fog is at half out to 0.93 of the explosion's radius, with 0.5, as it
	// was, to 0.55 - 0.59. The source alone would give 0.81 x the impact's radius (0.65 and 0.41 of the explosion's);
	// the air the blast throws outward carries the thin fog further.
	inline float ExplosionRadius(float a_gameRadius, float a_scale, float a_fallback)
	{
		return std::clamp(std::isfinite(a_gameRadius) ? a_gameRadius * a_scale : a_fallback, 40.0f, 600.0f);
	}

	// ---- the fog field around a body (EnableProfiler line "fog around player", WindField.cpp; WindSimCheck.cpp reads
	// the same regions on a simulated lane)

	// One region's q = rho / rho_eq (rho_eq = rho - delta where it is over 0.05, else q = 1, as CSColumn) and s (1 + the
	// relief texture, CSColumn's s - 1) over the texel centres inside it. Wall texels are left out (CSAdvectFog holds them
	// at rho 1, which would lift the means) and counted apart.
	struct TrailPart
	{
		int    n{ 0 };
		float  qMin{ 0.0f }, qMean{ 0.0f }, sMin{ 0.0f }, sMean{ 0.0f };
		double qSum{ 0.0 }, sSum{ 0.0 };
	};

	struct TrailSummary
	{
		TrailPart behind;   // 2R..6R behind the body along its heading, within R / 2 of its line: the lane's middle
		TrailPart ahead;    // R..3R from the body, within 45 degrees of straight ahead (WindSimCheck's front region)
		TrailPart side;     // 2R..6R behind, 3R..5R to either side: fog the body left alone, past the banks
		int       walls{ 0 };
	};

	inline constexpr float kTrailBehindNear = 2.0f, kTrailBehindFar = 6.0f, kTrailBehindHalfWidth = 0.5f;
	inline constexpr float kTrailAheadNear = 1.0f, kTrailAheadFar = 3.0f;
	inline constexpr float kTrailSideNear = 3.0f, kTrailSideFar = 5.0f;

	// a_w x a_h texels whose (0, 0) centre is at world (a_x0, a_y0), a_cell apart. a_fog(x, y, rho, delta) fills the fog
	// field's two channels, a_relief(x, y) returns s - 1, a_wall(x, y) true at a wall. The body stands at (a_bodyX,
	// a_bodyY), heading (a_dirX, a_dirY) (any length over 0), radius a_radius. O(a_w a_h).
	template <class Fog, class Relief, class Wall>
	TrailSummary SummarizeTrail(int a_w, int a_h, float a_x0, float a_y0, float a_cell, float a_bodyX, float a_bodyY, float a_dirX, float a_dirY,
		float a_radius, Fog&& a_fog, Relief&& a_relief, Wall&& a_wall)
	{
		TrailSummary out;
		const float  len = std::hypot(a_dirX, a_dirY);
		if (!(len > 1e-6f) || !(a_radius > 0.0f)) {
			return out;
		}
		const float fx = a_dirX / len, fy = a_dirY / len, R = a_radius;
		auto add = [](TrailPart& a_p, float a_q, float a_s) {
			a_p.qMin = a_p.n == 0 ? a_q : std::min(a_p.qMin, a_q);
			a_p.sMin = a_p.n == 0 ? a_s : std::min(a_p.sMin, a_s);
			a_p.qSum += a_q;
			a_p.sSum += a_s;
			++a_p.n;
		};
		for (int y = 0; y < a_h; ++y) {
			for (int x = 0; x < a_w; ++x) {
				const float dx = a_x0 + x * a_cell - a_bodyX;
				const float dy = a_y0 + y * a_cell - a_bodyY;
				const float along = dx * fx + dy * fy;   // over 0 ahead
				const float cross = std::fabs(dy * fx - dx * fy);
				const float dist = std::hypot(dx, dy);
				const bool  behind = -along >= kTrailBehindNear * R && -along <= kTrailBehindFar * R;
				TrailPart*  part = nullptr;
				if (behind && cross <= kTrailBehindHalfWidth * R) {
					part = &out.behind;
				} else if (along > 0.0f && cross <= along && dist >= kTrailAheadNear * R && dist <= kTrailAheadFar * R) {
					part = &out.ahead;
				} else if (behind && cross >= kTrailSideNear * R && cross <= kTrailSideFar * R) {
					part = &out.side;
				}
				if (!part) {
					continue;
				}
				if (a_wall(x, y)) {
					++out.walls;
					continue;
				}
				float rho = 1.0f, delta = 0.0f;
				a_fog(x, y, rho, delta);
				const float eq = rho - delta;
				add(*part, eq > 0.05f ? rho / eq : 1.0f, 1.0f + a_relief(x, y));
			}
		}
		for (TrailPart* p : { &out.behind, &out.ahead, &out.side }) {
			if (p->n > 0) {
				p->qMean = static_cast<float>(p->qSum / p->n);
				p->sMean = static_cast<float>(p->sSum / p->n);
			}
		}
		return out;
	}
}
