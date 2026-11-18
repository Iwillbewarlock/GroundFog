#pragma once

#include <array>
#include <cstdint>
#include <string>

// The grass probe's call bookkeeping for [Debug] GrassProbe = 2 (GrassProbe.cpp), free of D3D and of globals so the
// offline check can drive it with call sequences (tools/offline/GrassProbeCheck.cpp).
//
// A prepass "round" is one grass-range call on the prepass accumulator and the calls after it up to its group-9 call
// (probe1, DEBUG-LEDGER.md 2026-11-15: range 0x2B, grass, group 1, group 9, then group 4 from the Decals function). On a
// deferred frame the grass call of a kept round is held back (the stash) and made right after the round's group-9 call
// returns. The stash is never left behind silently:
//   - planned point: after the round's group-9 call returns;
//   - late point: before the next hooked call on the same accumulator, inside the prepass, with the main depth selected,
//     whatever its site (a round that closes without a group-9 call);
//   - dropped (logged once by the caller, mode 2 then acts as mode 1): another accumulator, another depth target or a
//     call outside the prepass comes first, or the fog's draw point / the frame end arrives with the call still held.
//     Outside the prepass the call is not made because Main_RenderDepth has returned and Community Shaders' Terrain
//     Blending has already blended and copied the prepass depth (TerrainBlending.cpp BlendPrepassDepths): the game never
//     makes that call there.
// Calls that arrive while another hooked call (or a replay) is running are passed on untouched (Inside()).
namespace GrassProbeDefer
{
	inline constexpr int kRounds = 2;            // prepass rounds copied and possibly deferred
	inline constexpr int kGroupLowAniso = 9;     // the planned replay point
	inline constexpr int kGroupDecals = 4;       // closes a round (the Decals function's call)

	// One hooked RenderBatches call as the game made it.
	struct Args
	{
		std::uintptr_t fn{ 0 };      // the function this site calls (the game's, or the next hook in the chain)
		void*          acc{ nullptr };
		std::uint32_t  first{ 0 };
		std::uint32_t  last{ 0 };
		std::uint32_t  flags{ 0 };
		int            group{ -1 };
		int            site{ -1 };
	};

	// What the tracker needs to know about a call besides its arguments.
	struct Call
	{
		int         site{ -1 };
		const void* acc{ nullptr };
		int         group{ -1 };
		bool        grass{ false };       // the grass range 0x5C000030..0x5C00005C
		bool        prepass{ false };     // inside Main_RenderDepth
		bool        mainDepth{ false };   // the state cache selects the main depth (kMAIN)
	};

	enum class RoundState : std::uint8_t
	{
		kNone,       // not on an armed frame, or not seen
		kNormal,     // made at its place, copied before and after
		kDeferred,   // held back, replay pending
		kAfterG9,    // made after the round's group-9 call (planned)
		kLate,       // made before a later call on the same accumulator (the round had no group-9 call)
		kDropped     // never made: see the header comment
	};

	inline const char* RoundStateText(RoundState a_s)
	{
		switch (a_s) {
		case RoundState::kNormal:
			return "normal";
		case RoundState::kDeferred:
			return "deferred, replay pending";
		case RoundState::kAfterG9:
			return "deferred, replayed after the group-9 call";
		case RoundState::kLate:
			return "deferred, replayed late (before a later call on the same accumulator)";
		case RoundState::kDropped:
			return "deferred, NOT replayed (dropped)";
		default:
			return "not copied";
		}
	}

	// What a deferral decision needs from outside (arming, setting, the render-state precheck).
	struct Gate
	{
		bool        armed{ false };        // this frame is a comparison frame
		bool        deferFrame{ false };   // and an even one in mode 2
		bool        modeAllows{ false };   // the setting reads 2 now
		const char* stateBlock{ nullptr }; // non-null: the last normal comparison's state precheck failed (the reason)
	};

	struct Stash
	{
		bool active{ false };
		int  round{ -1 };
		Args args{};
	};

	// Decision for one call before it is passed on.
	struct Before
	{
		bool drop{ false };         // the held call can no longer be made: dropped before this call
		bool replay{ false };       // make the held call now, before this one (late point)
		int  round{ -1 };           // this call is the grass call of kept round `round`
		bool extraRound{ false };   // a prepass grass call beyond kRounds (passed on untouched)
		bool otherAcc{ false };     // a prepass grass call on another accumulator than the first one
		int  g9Round{ -1 };         // this call is round g9Round's group-9 call (the planned replay point after it)
		int  decalsRound{ -1 };     // this call is the group-4 call right after round decalsRound's group-9 call
	};

	class Tracker
	{
	public:
		bool Inside() const { return m_depth > 0; }
		void Enter() { ++m_depth; }
		void Exit()
		{
			if (m_depth > 0) --m_depth;
		}

		Before OnCall(const Call& a_c)
		{
			Before b{};
			const int afterG9 = m_afterG9;
			m_afterG9 = -1;
			if (m_stash.active) {
				const bool inRound = m_open >= 0 && m_open == m_stash.round && a_c.prepass && a_c.mainDepth && a_c.acc == m_stash.args.acc &&
				                     a_c.group != -1 && a_c.group != kGroupDecals;
				if (!inRound) {
					if (a_c.prepass && a_c.mainDepth && a_c.acc == m_stash.args.acc) b.replay = true;
					else b.drop = true;
				}
			}
			if (m_open >= 0) {
				const bool inRound = a_c.prepass && a_c.acc == m_roundAcc && a_c.group != -1 && a_c.group != kGroupDecals;
				if (!inRound) {
					m_open = -1;
				} else if (a_c.group == kGroupLowAniso) {
					m_g9After[static_cast<std::size_t>(m_open)] = true;
					m_g9Site[static_cast<std::size_t>(m_open)] = a_c.site;
					b.g9Round = m_open;
					m_afterG9 = m_open;
					m_open = -1;
				}
			}
			if (afterG9 >= 0 && a_c.prepass && a_c.acc == m_roundAcc && a_c.group == kGroupDecals) {
				b.decalsRound = afterG9;
			}
			if (a_c.grass && a_c.prepass) {
				if (!m_roundAcc) m_roundAcc = a_c.acc;
				if (a_c.acc != m_roundAcc) {
					b.otherAcc = true;
				} else {
					const int r = m_rounds++;
					if (r < kRounds) {
						m_open = r;
						b.round = r;
					} else {
						b.extraRound = true;
					}
				}
			}
			return b;
		}

		// For kept round a_round of an armed deferred frame: holds the call back and returns null, or returns why not.
		template <class SiteCheck>
		const char* TryDefer(int a_round, const Gate& a_g, SiteCheck&& a_siteIntact, const Args& a_args)
		{
			if (a_round < 0 || a_round >= kRounds) return "not a kept round";
			if (!a_g.armed || !a_g.deferFrame) return "not a deferred comparison frame";
			if (m_off) return "mode 2 switched off earlier this session";
			if (!a_g.modeAllows) return "the setting is not 2";
			if (m_stash.active) return "another deferred call is pending";
			const auto r = static_cast<std::size_t>(a_round);
			if (!m_prevG9After[r]) return "last frame's round had no group-9 call after its grass call";
			if (a_g.stateBlock) return a_g.stateBlock;
			if (!a_siteIntact(a_args.site)) return "another plugin rewrote the grass call after us";
			if (!a_siteIntact(m_prevG9Site[r])) return "another plugin rewrote the group-9 call after us";
			m_stash = { true, a_round, a_args };
			m_state[r] = RoundState::kDeferred;
			return nullptr;
		}

		void MarkNormal(int a_round)
		{
			if (a_round >= 0 && a_round < kRounds) m_state[static_cast<std::size_t>(a_round)] = RoundState::kNormal;
		}

		// Takes the held call to make it now; a late point switches mode 2 off for the session.
		Stash Take(bool a_planned)
		{
			const Stash s = m_stash;
			m_stash.active = false;
			if (s.active && s.round >= 0 && s.round < kRounds) {
				m_state[static_cast<std::size_t>(s.round)] = a_planned ? RoundState::kAfterG9 : RoundState::kLate;
			}
			if (!a_planned) m_off = true;
			return s;
		}

		// The held call will not be made. Returns its round (-1 if nothing was held).
		int Drop()
		{
			if (!m_stash.active) return -1;
			const int r = m_stash.round;
			m_stash.active = false;
			if (r >= 0 && r < kRounds) m_state[static_cast<std::size_t>(r)] = RoundState::kDropped;
			m_off = true;
			return r;
		}

		// Main_RenderDepth returned: rounds close; a held call waits for the next call (OnCall drops it there).
		void PrepassEnd()
		{
			m_open = -1;
			m_afterG9 = -1;
		}

		// Frame end: returns the round of a call still held (dropped here) or -1; this frame's round structure becomes
		// the previous frame's.
		int EndFrame()
		{
			const int dropped = Drop();
			m_prevG9After = m_g9After;
			m_prevG9Site = m_g9Site;
			m_g9After.fill(false);
			m_g9Site.fill(-1);
			m_state.fill(RoundState::kNone);
			m_roundAcc = nullptr;
			m_rounds = 0;
			m_open = -1;
			m_afterG9 = -1;
			m_depth = 0;
			return dropped;
		}

		bool         StashActive() const { return m_stash.active; }
		int          StashRound() const { return m_stash.active ? m_stash.round : -1; }
		bool         Off() const { return m_off; }
		std::uint32_t Rounds() const { return m_rounds; }
		const void*  RoundAcc() const { return m_roundAcc; }
		RoundState   State(int a_round) const { return a_round >= 0 && a_round < kRounds ? m_state[static_cast<std::size_t>(a_round)] : RoundState::kNone; }
		bool         G9After(int a_round) const { return a_round >= 0 && a_round < kRounds && m_g9After[static_cast<std::size_t>(a_round)]; }

	private:
		Stash                             m_stash{};
		std::array<RoundState, kRounds>   m_state{};
		std::array<bool, kRounds>         m_g9After{};
		std::array<int, kRounds>          m_g9Site{ -1, -1 };
		std::array<bool, kRounds>         m_prevG9After{};
		std::array<int, kRounds>          m_prevG9Site{ -1, -1 };
		const void*                       m_roundAcc{ nullptr };
		std::uint32_t                     m_rounds{ 0 };
		int                               m_open{ -1 };
		int                               m_afterG9{ -1 };
		int                               m_depth{ 0 };
		bool                              m_off{ false };
	};

	// Where a normal comparison's render-state precheck was taken: the setting, and the player's place (an exterior's
	// worldspace, so walking across exterior cells keeps it; an interior's own cell). A different context makes the
	// precheck stale.
	struct Context
	{
		int         mode{ 0 };
		bool        exterior{ false };
		const void* place{ nullptr };

		bool operator==(const Context&) const = default;
	};

	// The render-state precheck of the last normal comparison, per kept round: a round may be deferred only if that round
	// was checked (state recorded at its grass call and after its group-9 call) and both sets were equal, in the same
	// context. Owns its reason texts (Block's pointer stays valid until the next Record or Invalidate).
	class Precheck
	{
	public:
		void Record(const Context& a_ctx, std::uint64_t a_index, const std::array<bool, kRounds>& a_checked, const std::array<bool, kRounds>& a_passed,
			const std::array<std::string, kRounds>& a_diffs)
		{
			m_have = true;
			m_ctx = a_ctx;
			m_index = a_index;
			m_checked = a_checked;
			m_passed = a_passed;
			for (int r = 0; r < kRounds; ++r) {
				const auto ri = static_cast<std::size_t>(r);
				m_reason[ri] = !a_checked[ri] ? "round " + std::to_string(r) + " was not checked on normal comparison #" + std::to_string(a_index) :
				               !a_passed[ri]  ? "state differs (round " + std::to_string(r) + ": " + a_diffs[ri] + ") on normal comparison #" + std::to_string(a_index) :
				                                std::string();
			}
		}

		// Drops a precheck taken in another context; true when it did (the caller logs it once).
		bool Invalidate(const Context& a_now)
		{
			if (!m_have || m_ctx == a_now) return false;
			m_have = false;
			return true;
		}

		// Null when round a_round may be deferred, else why not.
		const char* Block(int a_round) const
		{
			if (!m_have) return "state precheck not done (no normal comparison with recorded state since the start or the last context change)";
			if (a_round < 0 || a_round >= kRounds) return "not a kept round";
			const auto r = static_cast<std::size_t>(a_round);
			return m_checked[r] && m_passed[r] ? nullptr : m_reason[r].c_str();
		}

		bool           Have() const { return m_have; }
		std::uint64_t  Index() const { return m_index; }
		const Context& Ctx() const { return m_ctx; }

	private:
		bool                             m_have{ false };
		Context                          m_ctx{};
		std::uint64_t                    m_index{ 0 };
		std::array<bool, kRounds>        m_checked{};
		std::array<bool, kRounds>        m_passed{};
		std::array<std::string, kRounds> m_reason{};
	};

	// One hooked call, start to end. Env (GrassProbe.cpp in game, a recorder offline) provides:
	//   void Game(const Args&)                     make a call (the hooked one or a held one)
	//   void Seen(const Call&, const Before&)      record it, arm the frame (round 0), state at a group-4 call
	//   Gate GateFor(int round)                    arming, deferred frame, setting, state precheck
	//   bool SiteIntact(int site)                  the call at that site still points at our hook
	//   void Decided(int round, const char* whyNot) deferral decision of an armed kept round (null = held back)
	//   void BeforeGrass(int round) / AfterGrass(int round)   copies around a grass call made at its place
	//   void G9Returned(int round)                 the round's group-9 call returned (the replay point)
	//   void ReplayBegin(int round) / ReplayEnd(int round, bool planned, int site)   around a held call being made
	//   void Dropped(int round)                    a held call that will not be made
	template <class Env>
	void ReplayHeld(Tracker& a_t, bool a_planned, int a_site, Env& a_env)
	{
		const Stash s = a_t.Take(a_planned);
		if (!s.active) return;
		a_env.ReplayBegin(s.round);
		a_t.Enter();
		a_env.Game(s.args);
		a_t.Exit();
		a_env.ReplayEnd(s.round, a_planned, a_site);
	}

	template <class Env>
	void Dispatch(Tracker& a_t, const Call& a_c, const Args& a_a, Env& a_env)
	{
		if (a_t.Inside()) {
			a_t.Enter();
			a_env.Game(a_a);
			a_t.Exit();
			return;
		}
		const Before b = a_t.OnCall(a_c);
		if (b.drop) a_env.Dropped(a_t.Drop());
		if (b.replay) ReplayHeld(a_t, false, a_c.site, a_env);
		a_env.Seen(a_c, b);
		bool around = false;
		if (b.round >= 0) {
			const Gate g = a_env.GateFor(b.round);
			if (g.armed) {
				const char* why = a_t.TryDefer(b.round, g, [&](int a_site) { return a_env.SiteIntact(a_site); }, a_a);
				a_env.Decided(b.round, why);
				if (!why) return;   // made later, with these arguments and this site's function
				a_t.MarkNormal(b.round);
				around = true;
				a_env.BeforeGrass(b.round);
			}
		}
		a_t.Enter();
		a_env.Game(a_a);
		a_t.Exit();
		if (around) a_env.AfterGrass(b.round);
		if (b.g9Round >= 0) {
			a_env.G9Returned(b.g9Round);
			if (a_t.StashRound() == b.g9Round) ReplayHeld(a_t, true, a_c.site, a_env);
		}
	}
}
