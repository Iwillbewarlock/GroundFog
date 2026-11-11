#include "PCH.h"

#include "WindField.h"

#include "Settings.h"
#include "Terrain.h"
#include "WindSim.h"

#include <mutex>
#include <unordered_map>

namespace WindField
{
	namespace
	{
		using Clock = std::chrono::steady_clock;

		enum class Kind
		{
			kProjectile,
			kExplosion,
			kShout
		};

		struct Impact
		{
			Kind              kind;
			RE::NiPoint3      position;
			RE::NiPoint3      direction;  // horizontal part is used
			float             radius;
			Clock::time_point when;
			float             simTime{ 0.0f };  // field time the pulse started (set when moved into the active list)
			float             lastHit{ 0.0f };  // field time of the latest report merged into this impact
			float             heldSince{ -1.0f };  // field time a sustained hit reached the pulse's peak (-1 = not held)
		};

		struct ActorTrack
		{
			RE::NiPoint3      position;
			Clock::time_point when;
		};

		constexpr std::size_t kMaxQueued = 64;
		constexpr float       kUnitsPerMetre = 70.0f;
		constexpr float       kSqrtPi = 1.7724539f;
		constexpr float       kTwoPi = 6.2831853f;
		constexpr float       kSustain = 0.3f;       // share of the pulse's peak a sustained hit settles at
		constexpr float       kSustainEase = 0.5f;   // seconds to get there

		// An impact acts for BlastDuration seconds with a Hann-window pulse in time,
		//     w(t) = (1 - cos(2 pi t / T)) / T   for 0 <= t <= T,   integral over the pulse = 1:
		// it starts and ends at rest and peaks at T / 2. Rates are (setting) x w(t), so the settings are totals:
		// BlastDepth is the divergence integrated over the pulse at the centre (the fog there thins to about
		// exp(-BlastDepth) before it refills), BlastPush the speed the air gains along the hit direction.
		float Pulse(float a_age, float a_duration)
		{
			if (a_age < 0.0f || a_age > a_duration || a_duration <= 0.0f) {
				return 0.0f;
			}
			return (1.0f - std::cos(kTwoPi * a_age / a_duration)) / a_duration;
		}

		std::mutex          g_lock;  // impacts can arrive from other threads
		std::vector<Impact> g_queue;
		std::vector<Impact> g_active;
		WindSim             g_sim;
		Stats               g_stats{};
		bool                g_failed{ false };
		float               g_originX{ 0.0f };
		float               g_originY{ 0.0f };
		float               g_extent{ 0.0f };
		bool                g_haveGrid{ false };
		float               g_gustX{ 0.0f };
		float               g_gustY{ 0.0f };
		float               g_driftSpeed{ 0.0f };
		Clock::time_point   g_last{};
		std::unordered_map<RE::FormID, ActorTrack> g_tracks;
		std::atomic<bool>   g_resetRequested{ true };  // set from the message thread, acted on in Update
		float               g_simTime{ 0.0f };          // seconds the field has run (stops while paused)
		bool                g_loggedShoutMiss{ false };

		void Queue(Kind a_kind, const RE::NiPoint3& a_pos, const RE::NiPoint3& a_dir, float a_radius)
		{
			if (!Settings::enableWind || !std::isfinite(a_pos.x) || !std::isfinite(a_pos.y)) {
				return;
			}
			const std::scoped_lock lock(g_lock);
			if (g_queue.size() >= kMaxQueued) {
				g_queue.erase(g_queue.begin());
			}
			g_queue.push_back({ a_kind, a_pos, a_dir, a_radius, Clock::now() });
			++g_stats.impacts;
		}

		// Projectile::OnHit-style virtual (index 0xBD on the projectile vtables, signature as used by DTD's
		// MagicImpacts.cpp; DTD is GPL, only the facts are taken).
		template <class T>
		struct ProjectileHit
		{
			static void thunk(RE::Projectile* a_projectile, RE::TESObjectREFR* a_target, const RE::NiPoint3& a_position,
				const RE::NiPoint3& a_velocity, RE::hkpCollidable* a_collidable, std::int32_t a_arg6, std::uint32_t a_arg7)
			{
				if (a_projectile && a_projectile->GetProjectileRuntimeData().spell &&
					!a_projectile->GetProjectileRuntimeData().explosion) {  // an explosion reports itself
					Queue(Kind::kProjectile, a_position, a_velocity, Settings::blastRadius);
				}
				func(a_projectile, a_target, a_position, a_velocity, a_collidable, a_arg6, a_arg7);
			}
			static inline REL::Relocation<decltype(thunk)> func;
			static void Install()
			{
				REL::Relocation<std::uintptr_t> table{ T::VTABLE[0] };
				func = table.write_vfunc(0xBD, thunk);
			}
		};

		struct ExplosionInit
		{
			static void thunk(RE::Explosion* a_explosion)
			{
				func(a_explosion);
				if (!a_explosion) {
					return;
				}
				const float radius = a_explosion->GetExplosionRuntimeData().radius;
				const float r = std::clamp(std::isfinite(radius) ? radius * 0.5f : Settings::blastRadius, 40.0f, 600.0f);
				Queue(Kind::kExplosion, a_explosion->GetPosition(), RE::NiPoint3{ 0, 0, 0 }, r);
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		class CastSink final : public RE::BSTEventSink<RE::TESSpellCastEvent>
		{
		public:
			RE::BSEventNotifyControl ProcessEvent(const RE::TESSpellCastEvent* a_event,
				RE::BSTEventSource<RE::TESSpellCastEvent>*) override
			{
				if (!a_event || !a_event->object) {
					return RE::BSEventNotifyControl::kContinue;
				}
				auto* spell = RE::TESForm::LookupByID<RE::SpellItem>(a_event->spell);
				if (!spell) {
					if (!g_loggedShoutMiss) {
						g_loggedShoutMiss = true;
						logger::info("wind field: a cast event carried form {:08X} that is not a SpellItem; such casts are not seen", a_event->spell);
					}
					return RE::BSEventNotifyControl::kContinue;
				}
				if (spell->GetSpellType() != RE::MagicSystem::SpellType::kVoicePower) {
					return RE::BSEventNotifyControl::kContinue;
				}
				auto*       ref = a_event->object.get();
				const float yaw = ref->GetAngleZ();
				RE::NiPoint3 origin = ref->GetPosition();
				const RE::NiPoint3 dir{ std::sin(yaw), std::cos(yaw), 0.0f };
				origin.x += dir.x * 60.0f;
				origin.y += dir.y * 60.0f;
				Queue(Kind::kShout, origin, dir, Settings::blastRadius * 1.5f);
				return RE::BSEventNotifyControl::kContinue;
			}
		} g_castSink;

		bool EnsureCreated(ID3D11Device* a_device)
		{
			if (g_failed) {
				return false;
			}
			if (g_sim.Ready() && g_sim.Resolution() == Settings::simResolution) {
				return true;
			}
			std::string error;
			const auto  createStart = std::chrono::steady_clock::now();
			if (!g_sim.Create(a_device, Settings::simResolution, error)) {
				g_failed = true;
				g_stats.error = error;
				logger::error("wind field: compute setup failed: {}", error);
				return false;
			}
			g_stats.created = true;
			g_resetRequested = true;
			logger::info("wind field ready: {}x{} texels, created in {:.1f} ms on the render thread", Settings::simResolution, Settings::simResolution,
				std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - createStart).count());
			return true;
		}

		void FillActors(WindShaders::SimCB& a_cb, const RE::NiPoint3& a_centre, Clock::time_point a_now)
		{
			int  count = 0;
			auto add = [&](RE::Actor* a_actor) {
				if (!a_actor || count >= std::min(Settings::maxActors, WindShaders::kMaxDrags) || !a_actor->Get3D()) {
					return;
				}
				const auto pos = a_actor->GetPosition();
				const float dx = pos.x - a_centre.x;
				const float dy = pos.y - a_centre.y;
				if (dx * dx + dy * dy > (g_extent * 0.5f) * (g_extent * 0.5f)) {
					return;
				}
				auto& track = g_tracks[a_actor->GetFormID()];
				const float seconds = std::chrono::duration<float>(a_now - track.when).count();
				float vx = 0.0f;
				float vy = 0.0f;
				if (track.when.time_since_epoch().count() != 0 && seconds > 1e-3f && seconds < 0.5f) {
					vx = std::clamp((pos.x - track.position.x) / seconds, -1500.0f, 1500.0f);
					vy = std::clamp((pos.y - track.position.y) / seconds, -1500.0f, 1500.0f);
				}
				track = { pos, a_now };
				if (vx * vx + vy * vy < 25.0f) {
					return;  // standing still: no drag
				}
				a_cb.drags[count][0] = pos.x;
				a_cb.drags[count][1] = pos.y;
				a_cb.drags[count][2] = vx;
				a_cb.drags[count][3] = vy;
				++count;
			};
			add(RE::PlayerCharacter::GetSingleton());
			if (auto* lists = RE::ProcessLists::GetSingleton()) {
				for (auto& handle : lists->highActorHandles) {
					add(handle.get().get());
				}
			}
			a_cb.counts[0] = static_cast<float>(count);
			g_stats.lastActors = static_cast<std::uint32_t>(count);
			if (g_tracks.size() > 512) {
				g_tracks.clear();
			}
		}

		void FillBlasts(WindShaders::SimCB& a_cb, const RE::NiPoint3& a_centre)
		{
			std::vector<Impact> incoming;
			{
				const std::scoped_lock lock(g_lock);  // only the queue is shared with other threads
				incoming.swap(g_queue);
			}
			const float half = g_extent * 0.5f;
			const float duration = Settings::blastDuration;
			for (auto& impact : incoming) {
				if (std::fabs(impact.position.x - a_centre.x) > half || std::fabs(impact.position.y - a_centre.y) > half) {
					continue;  // outside the grid: would only take a slot from a nearby impact
				}
				impact.simTime = g_simTime;
				impact.lastHit = g_simTime;
				// A flame or beam touching the ground reports every frame: keep the nearby impact going instead of
				// starting a new pulse each frame (a pulse starts at rate 0, so restarting it would never push).
				// While the hits keep coming the impact is held at the pulse's peak, T / 2; when they stop it runs out.
				bool merged = false;
				for (auto& a : g_active) {
					const float dx = a.position.x - impact.position.x;
					const float dy = a.position.y - impact.position.y;
					if (a.kind == impact.kind && dx * dx + dy * dy < a.radius * a.radius && g_simTime - a.lastHit < 0.25f) {
						a.position = impact.position;
						a.direction = impact.direction;
						a.lastHit = g_simTime;
						if (g_simTime - a.simTime >= 0.5f * duration) {
							if (a.heldSince < 0.0f) {
								a.heldSince = g_simTime;   // reached the peak while still being hit
							}
							a.simTime = g_simTime - 0.5f * duration;
						}
						merged = true;
						break;
					}
				}
				if (!merged) {
					g_active.push_back(impact);
				}
			}
			std::erase_if(g_active, [duration](const Impact& a_i) { return g_simTime - a_i.simTime > duration; });
			// Over capacity: drop the impact that was hit longest ago (a flame still burning is the oldest entry but the
			// most recently hit; dropping it by age made it restart from rate 0 every frame; reviewer 2026-11-10).
			while (g_active.size() > static_cast<std::size_t>(WindShaders::kMaxBlasts)) {
				g_active.erase(std::min_element(g_active.begin(), g_active.end(),
					[](const Impact& a_l, const Impact& a_r) { return a_l.lastHit < a_r.lastHit; }));
			}
			int count = 0;
			for (const auto& impact : g_active) {
				// A sustained hit (flame, beam) eases from the pulse's peak down to kSustain of it over kSustainEase
				// seconds: held at the full peak it would blow at about BlastPush x 2 / T / (WindRelax +
				// VelocityDissipation), some 10 m/s with the defaults (reviewer's estimate, not measured).
				const float held = impact.heldSince >= 0.0f ? std::clamp((g_simTime - impact.heldSince) / kSustainEase, 0.0f, 1.0f) : 0.0f;
				const float pulse = Pulse(g_simTime - impact.simTime, duration) * (1.0f - (1.0f - kSustain) * held) *
				                    (impact.kind == Kind::kShout ? 1.5f : 1.0f);
				const float len = std::hypot(impact.direction.x, impact.direction.y);
				const float kick = len > 1e-3f ? Settings::blastPush * pulse / len : 0.0f;
				a_cb.blasts[count][0] = impact.position.x;
				a_cb.blasts[count][1] = impact.position.y;
				a_cb.blasts[count][2] = 1.0f / (impact.radius * impact.radius);
				a_cb.blasts[count][3] = Settings::blastDepth * pulse;
				a_cb.blastKicks[count][0] = impact.direction.x * kick;
				a_cb.blastKicks[count][1] = impact.direction.y * kick;
				++count;
			}
			a_cb.counts[1] = static_cast<float>(count);
			g_stats.lastBlasts = static_cast<std::uint32_t>(count);
		}
	}

	void Install()
	{
		ProjectileHit<RE::MissileProjectile>::Install();
		ProjectileHit<RE::BeamProjectile>::Install();
		ProjectileHit<RE::FlameProjectile>::Install();
		ProjectileHit<RE::ConeProjectile>::Install();
		ProjectileHit<RE::GrenadeProjectile>::Install();
		REL::Relocation<std::uintptr_t> table{ RE::Explosion::VTABLE[0] };
		ExplosionInit::func = table.write_vfunc(0xA2, ExplosionInit::thunk);
		logger::info("wind field: projectile, explosion hooks installed");
	}

	void RegisterEvents()
	{
		if (auto* source = RE::ScriptEventSourceHolder::GetSingleton()) {
			source->AddEventSink<RE::TESSpellCastEvent>(&g_castSink);
			logger::info("wind field: spell cast events registered (shouts)");
		}
	}

	bool Update(ID3D11Device* a_device, ID3D11DeviceContext* a_ctx, ID3D11ShaderResourceView* a_heights)
	{
		if (!Settings::enableWind || !EnsureCreated(a_device)) {
			return false;
		}

		const auto& grid = Terrain::GetGrid();
		const float extent = grid.cell * static_cast<float>(grid.size - 1);
		const float simCell = extent / static_cast<float>(Settings::simResolution);
		int         dx = 0;
		int         dy = 0;
		const bool  rebuilt = Terrain::ConsumeChange(dx, dy);
		const float ratio = grid.cell / simCell;
		const int   k = static_cast<int>(std::lround(ratio));
		const bool  wholeTexels = std::fabs(ratio - k) < 1e-3f;  // Settings::Load keeps this true
		if (g_resetRequested.exchange(false)) {
			g_tracks.clear();
			g_active.clear();
			g_simTime = 0.0f;
			g_sim.Reset(a_ctx);
			++g_stats.resets;
		} else if (rebuilt || g_extent != extent) {
			g_sim.Reset(a_ctx);
			++g_stats.resets;
		} else if (dx != 0 || dy != 0) {
			if (wholeTexels) {
				g_sim.Shift(a_ctx, dx * k, dy * k);  // whole height cells = whole sim texels
				++g_stats.shifts;
			} else {
				g_sim.Reset(a_ctx);  // only when the grid moved, never every frame
				++g_stats.resets;
			}
		}
		g_originX = grid.originX;
		g_originY = grid.originY;
		g_extent = extent;
		g_haveGrid = true;

		const auto now = Clock::now();
		float      dt = g_last.time_since_epoch().count() ? std::chrono::duration<float>(now - g_last).count() : 1.0f / 60.0f;
		g_last = now;
		dt = std::clamp(dt, 0.0f, 1.0f / 30.0f);
		if (auto* ui = RE::UI::GetSingleton(); ui && ui->GameIsPaused()) {
			return true;  // paused: keep the field as it is (no step, blast clocks stop)
		}
		g_simTime += dt;

		// Base wind from Sky (Dynamic Wind writes it; read only). Units of windSpeed and the angle convention are
		// not documented; ranges are logged by the self-test and WindScale maps the value to units/s.
		float windX = 0.0f;
		float windY = 0.0f;
		if (auto* sky = RE::Sky::GetSingleton()) {
			const float speed = std::isfinite(sky->windSpeed) ? sky->windSpeed : 0.0f;
			const float angle = std::isfinite(sky->windAngle) ? sky->windAngle : 0.0f;
			g_stats.windSpeedMin = std::min(g_stats.windSpeedMin, speed);
			g_stats.windSpeedMax = std::max(g_stats.windSpeedMax, speed);
			g_stats.windAngleMin = std::min(g_stats.windAngleMin, angle);
			g_stats.windAngleMax = std::max(g_stats.windAngleMax, angle);
			windX = std::sin(angle) * speed * Settings::windScale;
			windY = std::cos(angle) * speed * Settings::windScale;
		}
		g_gustX += windX * dt;
		g_gustY += windY * dt;
		const float baseSpeed = std::hypot(windX, windY);
		g_driftSpeed = baseSpeed;

		WindShaders::SimCB cb{};
		cb.grid[0] = grid.originX;
		cb.grid[1] = grid.originY;
		cb.grid[2] = simCell;
		cb.grid[3] = static_cast<float>(Settings::simResolution);
		cb.time[0] = dt;
		cb.time[1] = g_simTime;   // the clock BuildColumns passes too: the equilibrium pattern must be the same in both
		cb.time[2] = g_gustX;
		cb.time[3] = g_gustY;
		cb.wind[0] = windX;
		cb.wind[1] = windY;
		cb.wind[2] = Settings::windRelax;
		cb.wind[3] = Settings::vorticity;
		cb.gust[0] = Settings::gustAmount * baseSpeed + Settings::gustMinimum;
		cb.gust[1] = Settings::gustWavelength;
		cb.gust[2] = Settings::fogRefill;
		cb.gust[3] = Settings::velocityDissipation;
		cb.shift[3] = static_cast<float>(grid.size);
		cb.counts[2] = Settings::wallSlope;
		cb.counts[3] = Settings::wispAmount;
		cb.heightGrid[0] = grid.originX;
		cb.heightGrid[1] = grid.originY;
		cb.heightGrid[2] = grid.cell;
		cb.heightGrid[3] = Settings::enableWalls ? 1.0f : 0.0f;
		cb.dragParams[0] = Settings::actorRadius;
		cb.dragParams[1] = Settings::actorFollow;
		// ActorPush is the fog amount removed on a runner's path: the fog there thins to about exp(-ActorPush).
		// The shader's source is z * speed * Ricker(ActorRadius), whose integral along the path is
		// z * sqrt(pi) * R / 2 (WindShaders.h Ricker), hence the division.
		cb.dragParams[2] = Settings::actorPush / (0.5f * kSqrtPi * Settings::actorRadius);
		cb.dragParams[3] = Settings::actorPushMinSpeed;

		RE::NiPoint3 centre{ grid.originX + extent * 0.5f, grid.originY + extent * 0.5f, 0.0f };
		FillActors(cb, centre, now);
		FillBlasts(cb, centre);

		g_sim.Step(a_ctx, cb, Settings::enableWalls ? a_heights : nullptr, Settings::pressureIterations);
		++g_stats.steps;
		(void)kUnitsPerMetre;
		return true;
	}

	void Reset()
	{
		g_resetRequested = true;  // Update clears tracks and impacts on its own thread
		const std::scoped_lock lock(g_lock);
		g_queue.clear();
	}

	void BuildColumns(ID3D11DeviceContext* a_ctx, ID3D11ShaderResourceView* a_heights, ID3D11ShaderResourceView* a_place,
		std::array<float, 4> a_placeParams, std::array<float, 4> a_steamParams, std::array<float, 4> a_placeGrid,
		std::array<float, 4> a_look, ID3D11ShaderResourceView* a_coarse, std::array<float, 4> a_coarseGrid,
		std::array<float, 4> a_coarseParams)
	{
		if (!g_sim.Ready() || !g_haveGrid) {
			return;
		}
		g_sim.BuildColumns(a_ctx, a_heights, Settings::thickness, g_simTime, g_gustX, g_gustY, a_place, a_placeParams, a_steamParams, a_placeGrid, a_look, a_coarse, a_coarseGrid, a_coarseParams);
	}

	ID3D11ShaderResourceView* ColumnSrv() { return g_sim.Ready() ? g_sim.ColumnSrv() : nullptr; }
	ID3D11ShaderResourceView* UpperSrv() { return g_sim.Ready() ? g_sim.UpperSrv() : nullptr; }
	bool                      Ready() { return g_sim.Ready() && g_haveGrid && Settings::enableWind; }
	float                     OriginX() { return g_originX; }
	float                     OriginY() { return g_originY; }
	float                     Extent() { return g_extent; }
	float                     GustX() { return g_gustX; }
	float                     SimTime() { return g_simTime; }
	float                     GustY() { return g_gustY; }
	float                     DriftSpeed() { return g_driftSpeed; }
	const Stats&              GetStats() { return g_stats; }
}
