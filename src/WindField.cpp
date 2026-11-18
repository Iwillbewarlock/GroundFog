#include "PCH.h"

#include "WindField.h"

#include "GridMath.h"
#include "Settings.h"
#include "Terrain.h"
#include "WindRules.h"
#include "WindSim.h"

#include <DirectXPackedVector.h>

#include <array>
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
			bool              fire;       // a fire spell: boils fog off (WindRules::SetBlast)
			Clock::time_point when;
			float             simTime{ 0.0f };  // field time the pulse started (set when moved into the active list)
			float             lastHit{ 0.0f };  // field time of the latest report merged into this impact
			float             heldSince{ -1.0f };  // field time a sustained hit reached the pulse's peak (-1 = not held)
		};

		struct ActorTrack
		{
			RE::NiPoint3      position;
			Clock::time_point when;
			float             run{ 0.0f };               // straight run so far (units), this frame included: WakeDose
			float             headX{ 0.0f }, headY{ 0.0f };   // heading of the last frame (unit vector; 0 while still)
		};

		constexpr std::size_t kMaxQueued = 64;
		constexpr float       kUnitsPerMetre = 70.0f;
		constexpr std::size_t kMaxFireExplosions = 64;
		constexpr std::size_t kMaxBodyLogs = 12;

		std::mutex          g_lock;  // impacts can arrive from other threads
		std::vector<Impact> g_queue;
		std::vector<RE::FormID> g_fireExplosions;  // explosion forms seen on fire spells' projectiles (under g_lock)
		std::vector<RE::FormID> g_loggedBodies;    // render thread only
		std::vector<Impact> g_active;
		WindSim             g_sim;
		Stats               g_stats{};
		bool                g_failed{ false };
		int                 g_failedResolution{ 0 };   // the settings the failed setup was asked for (EnsureCreated)
		int                 g_failedSampling{ 0 };
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

		// Base wind fed to the field after [Wind] InputSmoothing, and the game's frame-to-frame changes (render thread).
		// The changes are measured on the wind vector in units/s (sin/cos of the angle x speed x WindScale), so they
		// do not depend on the undocumented units of Sky::windAngle.
		struct WindInput
		{
			bool  init{ false };    // smoothed value holds a game value
			float smoothX{ 0.0f };
			float smoothY{ 0.0f };
			bool  havePrev{ false };
			float prevX{ 0.0f };
			float prevY{ 0.0f };
			float prevSpeed{ 0.0f };
			// window since the last DescribeInput()
			int   frames{ 0 };
			float sumDv{ 0.0f };    // |raw vector change| per frame, units/s
			float maxDv{ 0.0f };
			float sumDs{ 0.0f };    // |Sky::windSpeed change| per frame
			float maxDs{ 0.0f };
			float sumV{ 0.0f };     // |raw vector|
			float sumSmoothDv{ 0.0f };   // |vector change| of what the field got
		};
		WindInput g_windIn{};

		// The player as FillActors saw it this frame (render thread), for the "fog around player" line.
		struct PlayerSample
		{
			bool  seen{ false };
			float x{ 0.0f }, y{ 0.0f }, vx{ 0.0f }, vy{ 0.0f }, radius{ 0.0f };
		};
		PlayerSample g_player{};

		// EnableProfiler: every 10 s the fog field around the player, read back without stalling the frame. Right after
		// the columns are built, a square of texels around the player (6.5 trail radii each way) is copied on the GPU
		// (CopySubresourceRegion) from the fog (RG32F), relief (R16F) and wall (R16F) textures into small staging
		// textures. From the second frame after, each frame tries Map with D3D11_MAP_FLAG_DO_NOT_WAIT and gives up the
		// try while the GPU has not done the copy (DXGI_ERROR_WAS_STILL_DRAWING); the line is logged the frame the three
		// maps succeed, with what the region looked like when it was copied. One request at a time; nothing is created
		// or copied while the profiler is off. Cost: one small copy per 10 s and a WindRules::SummarizeTrail pass over
		// at most (2 x 6.5 R / cell)^2 texels (28 x 28 for a person at 32-unit texels) once per 10 s.
		struct TrailProbe
		{
			ID3D11Texture2D*  fog{ nullptr };
			ID3D11Texture2D*  relief{ nullptr };
			ID3D11Texture2D*  wall{ nullptr };
			int               size{ 0 };          // texels a side of the staging textures
			bool              pending{ false };   // a copy waits to be read
			int               frames{ 0 };        // frames since that copy
			Clock::time_point lastRequest{};
			// what the copied square was: its (0, 0) texel centre in world units, the texel size, and the player then
			float             x0{ 0.0f }, y0{ 0.0f }, cell{ 0.0f }, bodyX{ 0.0f }, bodyY{ 0.0f }, dirX{ 0.0f }, dirY{ 0.0f }, radius{ 0.0f }, response{ 0.0f };
		};
		TrailProbe g_probe;

		constexpr float kProbeReach = 6.5f;          // trail radii each way from the player (the farthest region ends at 6 R)
		constexpr int   kProbeFirstTry = 2;          // frames after the copy before the first Map
		constexpr int   kProbeGiveUp = 120;          // frames after which a copy is dropped unread
		constexpr auto  kProbeEvery = std::chrono::seconds(10);

		void ReleaseProbe()
		{
			for (auto* t : { &g_probe.fog, &g_probe.relief, &g_probe.wall }) {
				if (*t) {
					(*t)->Release();
					*t = nullptr;
				}
			}
			g_probe.size = 0;
			g_probe.pending = false;
		}

		// Staging copies of a_src's format, a_size x a_size.
		bool MakeStaging(ID3D11Device* a_device, ID3D11Texture2D* a_src, int a_size, ID3D11Texture2D*& a_out)
		{
			D3D11_TEXTURE2D_DESC d{};
			a_src->GetDesc(&d);
			d.Width = d.Height = static_cast<UINT>(a_size);
			d.MipLevels = d.ArraySize = 1;
			d.SampleDesc = { 1, 0 };
			d.Usage = D3D11_USAGE_STAGING;
			d.BindFlags = 0;
			d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			d.MiscFlags = 0;
			return SUCCEEDED(a_device->CreateTexture2D(&d, nullptr, &a_out));
		}

		// Tries to read the pending copy; logs and clears it when the GPU is done with it (or after kProbeGiveUp frames).
		void ReadProbe(ID3D11DeviceContext* a_ctx)
		{
			if (++g_probe.frames < kProbeFirstTry) {
				return;
			}
			std::array<ID3D11Texture2D*, 3>          tex{ g_probe.fog, g_probe.relief, g_probe.wall };
			std::array<D3D11_MAPPED_SUBRESOURCE, 3> m{};
			int                                      mapped = 0;
			HRESULT                                  hr = S_OK;
			for (; mapped < 3; ++mapped) {
				hr = a_ctx->Map(tex[mapped], 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &m[mapped]);
				if (FAILED(hr)) {
					break;
				}
			}
			if (mapped < 3) {
				for (int i = 0; i < mapped; ++i) a_ctx->Unmap(tex[i], 0);
				if (hr == DXGI_ERROR_WAS_STILL_DRAWING && g_probe.frames < kProbeGiveUp) {
					return;   // try again next frame
				}
				g_probe.pending = false;
				logger::warn("fog around player: readback dropped after {} frames (hr {:08X})", g_probe.frames, static_cast<std::uint32_t>(hr));
				return;
			}
			const auto& p = g_probe;
			auto row = [&](int a_i, int a_y) { return static_cast<const std::uint8_t*>(m[a_i].pData) + static_cast<std::size_t>(a_y) * m[a_i].RowPitch; };
			const auto sum = WindRules::SummarizeTrail(
				p.size, p.size, p.x0, p.y0, p.cell, p.bodyX, p.bodyY, p.dirX, p.dirY, p.radius,
				[&](int a_x, int a_y, float& a_rho, float& a_delta) {
					const auto* f = reinterpret_cast<const float*>(row(0, a_y)) + 2 * a_x;
					a_rho = f[0];
					a_delta = f[1];
				},
				[&](int a_x, int a_y) { return DirectX::PackedVector::XMConvertHalfToFloat(reinterpret_cast<const std::uint16_t*>(row(1, a_y))[a_x]); },
				[&](int a_x, int a_y) { return DirectX::PackedVector::XMConvertHalfToFloat(reinterpret_cast<const std::uint16_t*>(row(2, a_y))[a_x]) > 0.5f; });
			for (int i = 0; i < 3; ++i) a_ctx->Unmap(tex[i], 0);
			g_probe.pending = false;
			const float heading = std::atan2(p.dirY, p.dirX) * 57.29578f;
			logger::info("fog around player (R {:.0f}, speed {:.0f}, heading {:.0f} deg, SurfaceResponse {:.2f}, read {} frames after the copy): "
						 "behind 2-6R within 0.5R n {} q min {:.3f} mean {:.3f} s min {:.3f} mean {:.3f}; "
						 "ahead R-3R within 45 deg n {} q min {:.3f} mean {:.3f} s min {:.3f} mean {:.3f}; "
						 "side 2-6R behind 3-5R out n {} q min {:.3f} mean {:.3f} s min {:.3f} mean {:.3f}; walls {}",
				p.radius, std::hypot(p.dirX, p.dirY), heading, p.response, p.frames,
				sum.behind.n, sum.behind.qMin, sum.behind.qMean, sum.behind.sMin, sum.behind.sMean,
				sum.ahead.n, sum.ahead.qMin, sum.ahead.qMean, sum.ahead.sMin, sum.ahead.sMean,
				sum.side.n, sum.side.qMin, sum.side.qMean, sum.side.sMin, sum.side.sMean, sum.walls);
		}

		// Called after BuildColumns while the profiler is on: reads a pending copy, or every kProbeEvery starts one.
		void UpdateProbe(ID3D11DeviceContext* a_ctx)
		{
			if (g_probe.pending) {
				ReadProbe(a_ctx);
				return;
			}
			const auto now = Clock::now();
			if (now - g_probe.lastRequest < kProbeEvery) {
				return;
			}
			g_probe.lastRequest = now;
			const float speed = std::hypot(g_player.vx, g_player.vy);
			if (!g_player.seen || !(speed >= WindRules::kStillSpeed)) {
				logger::info("fog around player: not read (seen {}, speed {:.0f}: no heading while standing)", g_player.seen, speed);
				return;
			}
			const int   res = g_sim.Resolution();
			const float cell = g_extent / static_cast<float>(res);
			const float radius = std::max(g_player.radius, 1.0f);
			const int   half = static_cast<int>(std::ceil(kProbeReach * radius / cell)) + 1;
			const int   size = std::min(2 * half, res);
			if (g_probe.size != size) {
				ReleaseProbe();
				ID3D11Device* device = nullptr;
				a_ctx->GetDevice(&device);
				const bool ok = device && MakeStaging(device, g_sim.FogTexture(), size, g_probe.fog) && MakeStaging(device, g_sim.ReliefTexture(), size, g_probe.relief) &&
				                MakeStaging(device, g_sim.WallTexture(), size, g_probe.wall);
				if (device) device->Release();
				if (!ok) {
					ReleaseProbe();
					logger::warn("fog around player: staging textures ({} x {}) could not be created", size, size);
					return;
				}
				g_probe.size = size;
			}
			const int px = static_cast<int>(std::floor((g_player.x - g_originX) / cell));
			const int py = static_cast<int>(std::floor((g_player.y - g_originY) / cell));
			const int x0 = std::clamp(px - half, 0, res - size);
			const int y0 = std::clamp(py - half, 0, res - size);
			const D3D11_BOX box{ static_cast<UINT>(x0), static_cast<UINT>(y0), 0, static_cast<UINT>(x0 + size), static_cast<UINT>(y0 + size), 1 };
			a_ctx->CopySubresourceRegion(g_probe.fog, 0, 0, 0, 0, g_sim.FogTexture(), 0, &box);
			a_ctx->CopySubresourceRegion(g_probe.relief, 0, 0, 0, 0, g_sim.ReliefTexture(), 0, &box);
			a_ctx->CopySubresourceRegion(g_probe.wall, 0, 0, 0, 0, g_sim.WallTexture(), 0, &box);
			g_probe.pending = true;
			g_probe.frames = 0;
			g_probe.x0 = g_originX + (static_cast<float>(x0) + 0.5f) * cell;
			g_probe.y0 = g_originY + (static_cast<float>(y0) + 0.5f) * cell;
			g_probe.cell = cell;
			g_probe.bodyX = g_player.x;
			g_probe.bodyY = g_player.y;
			g_probe.dirX = g_player.vx;
			g_probe.dirY = g_player.vy;
			g_probe.radius = radius;
			g_probe.response = Settings::surfaceResponse;
		}

		void Queue(Kind a_kind, const RE::NiPoint3& a_pos, const RE::NiPoint3& a_dir, float a_radius, bool a_fire)
		{
			if (!Settings::enableWind || !std::isfinite(a_pos.x) || !std::isfinite(a_pos.y)) {
				return;
			}
			const std::scoped_lock lock(g_lock);
			if (g_queue.size() >= kMaxQueued) {
				g_queue.erase(g_queue.begin());
			}
			g_queue.push_back({ a_kind, a_pos, a_dir, a_radius, a_fire, Clock::now() });
			++g_stats.impacts;
		}

		// Fire: the spell's costliest effect is resisted by fire resistance (Flames, Firebolt, Fireball, fire breath,
		// fire runes, and a mod's spells built the same way). A spell whose costliest effect is something else is
		// not seen as fire.
		bool IsFire(const RE::MagicItem* a_spell)
		{
			if (!a_spell) {
				return false;
			}
			const auto* effect = a_spell->GetCostliestEffectItem();
			return effect && effect->baseEffect && effect->baseEffect->data.resistVariable == RE::ActorValue::kResistFire;
		}

		// An explosion does not say which spell made it. Two ways it is known to be fire: its own form carries a fire
		// enchantment, or a fire spell's projectile carrying that explosion form has hit something (the hit comes
		// before the explosion it sets off, so the form is known from the first one on). Not covered: explosions
		// from no spell (traps, oil) whose form has no fire enchantment.
		void RememberFireExplosion(const RE::BGSExplosion* a_form)
		{
			if (!a_form) {
				return;
			}
			const std::scoped_lock lock(g_lock);
			const auto             id = a_form->GetFormID();
			if (std::find(g_fireExplosions.begin(), g_fireExplosions.end(), id) == g_fireExplosions.end() && g_fireExplosions.size() < kMaxFireExplosions) {
				g_fireExplosions.push_back(id);
			}
		}

		bool IsFireExplosion(RE::Explosion* a_explosion)
		{
			const auto* base = a_explosion->GetBaseObject();
			const auto* form = base ? base->As<RE::BGSExplosion>() : nullptr;
			if (!form) {
				return false;
			}
			if (IsFire(form->formEnchanting)) {
				return true;
			}
			const std::scoped_lock lock(g_lock);
			return std::find(g_fireExplosions.begin(), g_fireExplosions.end(), form->GetFormID()) != g_fireExplosions.end();
		}

		// Projectile::OnHit-style virtual (index 0xBD on the projectile vtables, signature as used by DTD's
		// MagicImpacts.cpp; DTD is GPL, only the facts are taken).
		template <class T>
		struct ProjectileHit
		{
			static void thunk(RE::Projectile* a_projectile, RE::TESObjectREFR* a_target, const RE::NiPoint3& a_position,
				const RE::NiPoint3& a_velocity, RE::hkpCollidable* a_collidable, std::int32_t a_arg6, std::uint32_t a_arg7)
			{
				if (a_projectile && a_projectile->GetProjectileRuntimeData().spell) {
					const auto& data = a_projectile->GetProjectileRuntimeData();
					const bool  fire = IsFire(data.spell);
					if (!data.explosion) {
						Queue(Kind::kProjectile, a_position, a_velocity, Settings::blastRadius, fire);
					} else if (fire) {
						RememberFireExplosion(data.explosion);  // the explosion reports itself (ExplosionInit)
					}
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
				const float r = WindRules::ExplosionRadius(a_explosion->GetExplosionRuntimeData().radius, Settings::explosionRadiusScale, Settings::blastRadius);
				Queue(Kind::kExplosion, a_explosion->GetPosition(), RE::NiPoint3{ 0, 0, 0 }, r, IsFireExplosion(a_explosion));
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
				Queue(Kind::kShout, origin, dir, Settings::blastRadius * 1.5f, IsFire(spell));
				return RE::BSEventNotifyControl::kContinue;
			}
		} g_castSink;

		bool EnsureCreated(ID3D11Device* a_device)
		{
			if (g_failed && g_failedResolution == Settings::simResolution && g_failedSampling == Settings::fogSampling) {
				return false;   // the same settings that failed; a changed Resolution or FogSampling is tried again
			}
			g_failed = false;
			if (g_sim.Ready() && g_sim.Resolution() == Settings::simResolution && g_sim.FogSampling() == Settings::fogSampling) {
				return true;
			}
			std::string error;
			const auto  createStart = std::chrono::steady_clock::now();
			if (!g_sim.Create(a_device, Settings::simResolution, error, Settings::fogSampling)) {
				g_failed = true;
				g_failedResolution = Settings::simResolution;
				g_failedSampling = Settings::fogSampling;
				g_stats.error = error;
				logger::error("wind field: compute setup failed: {}", error);
				return false;
			}
			g_stats.created = true;
			g_resetRequested = true;
			logger::info("wind field ready: {}x{} texels, FogSampling {}, created in {:.1f} ms on the render thread", Settings::simResolution, Settings::simResolution, g_sim.FogSampling(),
				std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - createStart).count());
			return true;
		}

		// The player's bounds once, and each kind of body wider than a person's lane the first time it moves in the
		// field: what the game reports for a body's size is not documented (whether the actor's scale is in it), so
		// the log shows what the lanes are sized from.
		void LogBody(RE::Actor* a_actor, const RE::NiPoint3& a_min, const RE::NiPoint3& a_max, float a_width)
		{
			if (g_loggedBodies.size() >= kMaxBodyLogs || (!a_actor->IsPlayerRef() && !(a_width > WindRules::kPersonLane))) {
				return;
			}
			const auto* base = a_actor->GetBaseObject();
			const auto  id = base ? base->GetFormID() : a_actor->GetFormID();
			if (std::find(g_loggedBodies.begin(), g_loggedBodies.end(), id) != g_loggedBodies.end()) {
				return;
			}
			g_loggedBodies.push_back(id);
			logger::info("wind field: body '{}' ({:08X}) bounds {:.0f} x {:.0f} x {:.0f} units, scale {:.2f}: width {:.0f}, trail radius {:.0f}, depth x {:.2f}",
				a_actor->GetName(), id, a_max.x - a_min.x, a_max.y - a_min.y, a_max.z - a_min.z, a_actor->GetScale(), a_width,
				WindRules::DragRadius(Settings::actorRadius, a_width), std::sqrt(WindRules::BodyScale(a_width)));
		}

		// The moving actors inside the field's circle, the nearest to the player first, as many as MaxActors (and the
		// shader) take. They used to be taken in the order of the game's list until the slots were full: over a field
		// 2.25 times the area a far actor could take the slot of one running next to the player. Every actor in the
		// circle is tracked (its speed comes from its last position), also those that get no slot this frame.
		// Left out (WindRules::Disturbs): a rider (the mount is in the list too and stirs the fog for both), anything
		// standing still, and anything with its feet above the layer's reach (flying, on a bridge or a wall).
		void FillActors(WindShaders::SimCB& a_cb, const RE::NiPoint3& a_centre, Clock::time_point a_now)
		{
			static std::vector<WindRules::Body> moving;   // render thread only; kept to save the allocations
			static std::vector<float>           dist2;    // squared horizontal distance to the player
			static std::vector<std::pair<float, float>> wake;   // per row: distance moved this frame, straight run so far (WakeDose)
			moving.clear();
			dist2.clear();
			wake.clear();
			auto*              player = RE::PlayerCharacter::GetSingleton();
			const RE::NiPoint3 playerPos = player ? player->GetPosition() : a_centre;
			g_player.seen = false;
			// The player as the field sees it, for the profiler line below (-1: no row in the list).
			bool           playerSeen = false;
			float          playerSpeed = 0.0f;
			bool           playerReach = false;
			bool           playerRider = false;
			std::ptrdiff_t playerRow = -1;
			auto add = [&](RE::Actor* a_actor) {
				if (!a_actor || !a_actor->Get3D()) {
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
				// Wake evaporation (WindShaders.h WakeDose): the path moved since the last frame, in real time, and the straight
				// run it extends; a stop, a gap in the tracking or a turn over 30 degrees in one frame starts a new run.
				const float speedNow = std::hypot(vx, vy);
				const float step = speedNow * (speedNow > 0.0f ? seconds : 0.0f);
				const bool  straight = speedNow > 0.0f && track.run > 0.0f && vx * track.headX + vy * track.headY >= WindRules::kWakeTurnCos * speedNow;
				const float run = speedNow > 0.0f ? (straight ? track.run : 0.0f) + step : 0.0f;
				// The same actor met again in the same frame (seconds ~ 0: e.g. the player also in the high list) keeps its run.
				const bool again = track.when.time_since_epoch().count() != 0 && seconds <= 1e-3f;
				if (again) {
					track.position = pos;
					track.when = a_now;
				} else {
					track = { pos, a_now, run, speedNow > 0.0f ? vx / speedNow : 0.0f, speedNow > 0.0f ? vy / speedNow : 0.0f };
				}
				WindRules::Body body{ pos.x, pos.y, pos.z, vx, vy, 0.0f, a_actor->IsOnMount(), false, 0.0f };
				body.groundKnown = Terrain::HeightAt(pos.x, pos.y, body.ground);
				const bool isPlayer = !playerSeen && a_actor == player;
				if (isPlayer) {
					playerSeen = true;
					playerSpeed = std::hypot(vx, vy);
					playerReach = WindRules::WithinLayer(pos.z, body.groundKnown, body.ground, Settings::thickness);
					playerRider = body.rider;
					g_player = { true, pos.x, pos.y, vx, vy, Settings::actorRadius };
				}
				if (!WindRules::Disturbs(body, Settings::thickness)) {
					return;
				}
				const auto boundMin = a_actor->GetBoundMin();
				const auto boundMax = a_actor->GetBoundMax();
				body.width = WindRules::BodyWidth(boundMin.x, boundMin.y, boundMax.x, boundMax.y);
				LogBody(a_actor, boundMin, boundMax, body.width);
				if (isPlayer) {
					playerRow = static_cast<std::ptrdiff_t>(moving.size());
					g_player.radius = WindRules::DragRadius(Settings::actorRadius, body.width);   // the trail radius SetDrag gives it
				}
				moving.push_back(body);
				wake.emplace_back(step, run);
				dist2.push_back((pos.x - playerPos.x) * (pos.x - playerPos.x) + (pos.y - playerPos.y) * (pos.y - playerPos.y));
			};
			add(player);   // first, at distance 0: keeps the first slot
			if (auto* lists = RE::ProcessLists::GetSingleton()) {
				for (auto& handle : lists->highActorHandles) {
					add(handle.get().get());
				}
			}
			const auto slots = static_cast<std::size_t>(std::clamp(Settings::maxActors, 0, WindShaders::kMaxDrags));
			int        count = 0;
			int        playerSlot = -1;
			for (const std::size_t i : GridMath::NearestFirst(dist2, slots)) {
				if (static_cast<std::ptrdiff_t>(i) == playerRow) {
					playerSlot = count;
				}
				WindRules::SetDrag(a_cb, count, moving[i], Settings::actorRadius, Settings::actorPush, wake[i].first, wake[i].second);
				++count;
			}
			a_cb.counts[0] = static_cast<float>(count);
			g_stats.lastActors = static_cast<std::uint32_t>(count);
			// EnableProfiler: every 5 s whether the field looked at the player this frame (seen false: no 3D or outside
			// the field's circle, so the other values are only their defaults), its speed as the field measures it, the
			// share of the push the shader gives at that speed (WindRules::DragGate with this frame's DragParams), its
			// slot (-1: none, e.g. standing, riding or above the reach), whether its feet are within the layer's reach and
			// whether it rides (a rider gets no row; its mount does). Diagnosis of "the fog does not react when I move"
			// (user test 1, 2026-11-14).
			static Clock::time_point lastPlayerLog{};
			if (Settings::profiler && a_now - lastPlayerLog >= std::chrono::seconds(5)) {
				lastPlayerLog = a_now;
				logger::info("player seen {} speed {:.0f} gate {:.2f} slot {} reach-ok {} rider {}", playerSeen, playerSpeed,
					WindRules::DragGate(playerSpeed, a_cb.dragParams[0], a_cb.dragParams[3]), playerSlot, playerReach, playerRider);
			}
			if (g_tracks.size() > 512) {
				g_tracks.clear();
			}
		}

		// Fires (ini [Wind] FireClear > 0): campfires, braziers, forges and the like found round the player every
		// kFireScanFrames frames by their model's name, each clearing the fog out to a little past its model's edge.
		// After roxorvic's Community Shaders fog (skyrim-community-shaders fluid-field/fog-v3, GroundTiles.cpp FIRES and
		// HeatReach, AirSources.cpp fires; GPL-3.0, used with permission). Render thread only.
		using Fire = WindRules::Heat;
		std::vector<Fire>     g_fires;
		int                   g_fireScanIn{ 0 };
		std::size_t           g_firesLogged{ static_cast<std::size_t>(-1) };
		constexpr int         kFireScanFrames = 300;   // fires do not move; 90 frames with a 9000-unit scan cost up to 7.45 ms each time in game (2026-11-17)
		std::unordered_map<RE::FormID, bool> g_fireBase;   // base form -> its model is a fire (the path test is the costly part)
		constexpr std::array<std::string_view, 9> kFireModels{ "campfire", "firepit", "fireplace", "brazier", "forge", "smelter", "hearth", "fxfire", "fire0" };
		constexpr float       kFireMargin = 24.0f;   // past the model's half width
		constexpr float       kFireLeast = 40.0f;
		constexpr float       kFireMost = 400.0f;

		float FireReach(RE::TESObjectREFR& a_ref)
		{
			const auto* base = a_ref.GetBaseObject();
			if (!base) return 0.0f;
			auto known = g_fireBase.find(base->GetFormID());
			if (known == g_fireBase.end()) {
				const auto* model = base->As<RE::TESModel>();
				bool        fire = false;
				if (model && model->GetModel()) {
					std::string path = model->GetModel();
					std::transform(path.begin(), path.end(), path.begin(), [](unsigned char a_c) { return static_cast<char>(std::tolower(a_c)); });
					fire = std::any_of(kFireModels.begin(), kFireModels.end(), [&](std::string_view a_part) { return path.find(a_part) != std::string::npos; });
				}
				known = g_fireBase.emplace(base->GetFormID(), fire).first;
			}
			if (!known->second) return 0.0f;
			const auto size = a_ref.GetBoundMax() - a_ref.GetBoundMin();
			return std::clamp(0.5f * std::max(size.x, size.y) * a_ref.GetScale() + kFireMargin, kFireLeast, kFireMost);
		}

		bool ScanFires()   // true on the frames it scans
		{
			if (g_fireScanIn-- > 0) return false;
			g_fireScanIn = kFireScanFrames;
			g_fires.clear();
			const auto scanStart = Clock::now();
			auto*      tes = RE::TES::GetSingleton();
			auto* player = RE::PlayerCharacter::GetSingleton();
			if (!tes || !player || tes->interiorCell) return true;
			tes->ForEachReferenceInRange(player, g_extent * 0.5f, [](RE::TESObjectREFR* a_ref) {
				if (a_ref && !a_ref->IsDisabled()) {
					if (const float reach = FireReach(*a_ref); reach > 0.0f) {
						// round its model's middle, not its origin (a forge's is at one end, its pit at the other)
						const auto*        node = a_ref->Get3D();
						const RE::NiPoint3 c = node ? node->worldBound.center : a_ref->GetPosition();
						g_fires.push_back({ c.x, c.y, a_ref->GetPositionZ(), reach, 0.0f });
					}
				}
				return RE::BSContainer::ForEachResult::kContinue;
			});
			if (g_fires.size() != g_firesLogged) {
				g_firesLogged = g_fires.size();
				logger::info("wind field: {} fires round the player (scan {:.2f} ms){}", g_fires.size(),
					std::chrono::duration<double, std::milli>(Clock::now() - scanStart).count(),
					g_fires.empty() ? std::string() : std::format(", the first at ({:.0f}, {:.0f}, {:.0f}) clearing {:.0f}", g_fires[0].x, g_fires[0].y, g_fires[0].z, g_fires[0].radius));
			}
			return true;
		}

		// Fire spells' burning ground (Wall of Flames and the like): hazards whose spell is fire, found every
		// kHazardScanFrames frames (they come and go within seconds) and followed by handle every frame; and torches
		// held by the player and the actors near it ([Wind] TorchClear, TorchReach), a faint clearing round the one
		// holding it. After NearMidnightNow's Dynamic Terrain Deformation (HeatSources.cpp IsHot and ForActor;
		// GPL-3.0, used with permission 2026-11-17). Render thread only.
		std::vector<RE::ObjectRefHandle>     g_hazards;
		int                                  g_hazardScanIn{ 0 };
		std::size_t                          g_hazardsLogged{ static_cast<std::size_t>(-1) };
		constexpr int                        kHazardScanFrames = 30;
		std::unordered_map<RE::FormID, bool> g_hazardBase;   // hazard base form -> its spell is fire

		bool IsFireHazard(const RE::TESBoundObject* a_base)
		{
			if (!a_base || a_base->GetFormType() != RE::FormType::Hazard) return false;
			auto known = g_hazardBase.find(a_base->GetFormID());
			if (known == g_hazardBase.end()) {
				const auto* hazard = a_base->As<RE::BGSHazard>();
				known = g_hazardBase.emplace(a_base->GetFormID(), hazard && IsFire(hazard->data.spell)).first;
			}
			return known->second;
		}

		void ScanHazards()
		{
			if (g_hazardScanIn-- > 0) return;
			g_hazardScanIn = kHazardScanFrames;
			g_hazards.clear();
			const auto scanStart = Clock::now();
			auto*      tes = RE::TES::GetSingleton();
			auto*      player = RE::PlayerCharacter::GetSingleton();
			if (!tes || !player || tes->interiorCell) return;
			float firstRadius = 0.0f;
			tes->ForEachReferenceInRange(player, g_extent * 0.5f, [&firstRadius](RE::TESObjectREFR* a_ref) {
				if (a_ref && !a_ref->IsDisabled() && !a_ref->IsMarkedForDeletion() && IsFireHazard(a_ref->GetBaseObject())) {
					if (g_hazards.empty()) {
						if (auto* placed = a_ref->As<RE::Hazard>()) firstRadius = placed->GetHazardRuntimeData().radius;
					}
					g_hazards.push_back(a_ref->CreateRefHandle());
				}
				return RE::BSContainer::ForEachResult::kContinue;
			});
			// logged when hazards appear or all go out (not at every count change while a flame wall is being drawn)
			if (g_hazardsLogged == static_cast<std::size_t>(-1) || (g_hazards.size() == 0) != (g_hazardsLogged == 0)) {
				g_hazardsLogged = g_hazards.size();
				logger::info("wind field: {} burning hazards round the player (scan {:.2f} ms){}", g_hazards.size(),
					std::chrono::duration<double, std::milli>(Clock::now() - scanStart).count(),
					g_hazards.empty() ? std::string() : std::format(", the first's own radius {:.0f}", firstRadius));
			}
		}

		// The player and the actors near it holding a torch (a carried Light in the left hand), at their feet.
		template <class F>
		void ForEachTorch(F&& a_add)
		{
			auto* tes = RE::TES::GetSingleton();
			if (!tes || tes->interiorCell) return;   // as the fires and hazards
			auto add = [&](RE::Actor* a_actor) {
				// not a swimmer (the torch is out of sight; Carbon review 2026-11-17), nor one disabled or being deleted
				if (!a_actor || !a_actor->Is3DLoaded() || a_actor->IsDead() || a_actor->IsDisabled() || a_actor->IsMarkedForDeletion() ||
					a_actor->AsActorState()->IsSwimming())
					return;
				const auto* held = a_actor->GetEquippedObject(true);
				if (!held || held->GetFormType() != RE::FormType::Light) return;
				const auto p = a_actor->GetPosition();
				a_add(Fire{ p.x, p.y, p.z, Settings::torchReach, Settings::torchClear });
			};
			add(RE::PlayerCharacter::GetSingleton());
			if (auto* lists = RE::ProcessLists::GetSingleton()) {
				for (auto& handle : lists->highActorHandles) {
					add(handle.get().get());
				}
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
				// Above the layer's reach (a spell hitting high on a cliff or a wall, a dragon shouting in the air):
				// nothing for the fog below. Here and not in Queue: the height grid belongs to this thread.
				float      ground = 0.0f;
				const bool groundKnown = Terrain::HeightAt(impact.position.x, impact.position.y, ground);
				if (!WindRules::WithinLayer(impact.position.z, groundKnown, ground, Settings::thickness)) {
					continue;
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
					if (a.kind == impact.kind && a.fire == impact.fire && dx * dx + dy * dy < a.radius * a.radius && g_simTime - a.lastHit < 0.25f) {
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
				// A sustained hit (flame, beam) eases from the pulse's peak down (WindRules::Sustained). A shout's
				// 1.5 is on its kick only: on the depth too it asked for a rim of 2.25, over the fog's cap of 2.
				const float rate = WindRules::Pulse(g_simTime - impact.simTime, duration) *
				                   WindRules::Sustained(impact.heldSince >= 0.0f ? g_simTime - impact.heldSince : -1.0f);
				const WindRules::Blast blast{ impact.position.x, impact.position.y, impact.direction.x, impact.direction.y, impact.radius, impact.fire,
					impact.kind == Kind::kShout };
				WindRules::SetBlast(a_cb, count, blast, rate, Settings::blastDepth, Settings::blastPush, Settings::fireEvaporate);
				++count;
			}
			// Fires, burning hazards and held torches take the slots the impacts leave, nearest first: a steady fire hit at
			// its rate (FireClear, TorchClear), no push of its own beyond the blast's gentle divergence (the edge churns),
			// boiling the fog off at FireEvaporate x that rate a second.
			if (Settings::fireClear > 0.0f || Settings::torchClear > 0.0f) {
				std::vector<Fire> near;
				auto consider = [&](const Fire& a_f) {
					if (std::fabs(a_f.x - a_centre.x) > half || std::fabs(a_f.y - a_centre.y) > half) return;
					// a brazier on a wall or a bridge high over the ground leaves the fog below alone (as the impacts above)
					float      ground = 0.0f;
					const bool known = Terrain::HeightAt(a_f.x, a_f.y, ground);
					if (!WindRules::WithinLayer(a_f.z, known, ground, Settings::thickness)) return;
					near.push_back(a_f);
				};
				if (Settings::fireClear > 0.0f) {
					if (!ScanFires()) ScanHazards();   // never both scans in one frame (Carbon review 2026-11-17)
					for (const auto& f : g_fires) {
						consider({ f.x, f.y, f.z, f.radius, Settings::fireClear });
					}
					for (const auto& h : g_hazards) {
						const auto ref = h.get();   // gone when the hazard burnt out
						if (!ref || ref->IsDisabled() || ref->IsMarkedForDeletion()) continue;
						// round its 3D's bounding sphere, as DTD's ReachFor; read each frame, as the 3D may load after the scan
						const auto* node = ref->Get3D();
						const auto  p = ref->GetPosition();
						consider({ p.x, p.y, p.z, std::clamp(node ? node->worldBound.radius : 0.0f, kFireLeast, kFireMost), Settings::fireClear });
					}
				}
				if (Settings::torchClear > 0.0f) {
					ForEachTorch(consider);
				}
				std::sort(near.begin(), near.end(), [&](const Fire& a_l, const Fire& a_r) {
					return std::hypot(a_l.x - a_centre.x, a_l.y - a_centre.y) < std::hypot(a_r.x - a_centre.x, a_r.y - a_centre.y);
				});
				// one slot per spot, the stronger source keeping it (WindRules::PickHeat)
				for (const auto& f : WindRules::PickHeat(near, static_cast<std::size_t>(std::max(WindShaders::kMaxBlasts - count, 0)))) {
					const WindRules::Blast blast{ f.x, f.y, 0.0f, 0.0f, f.radius, true, false };
					WindRules::SetBlast(a_cb, count, blast, f.rate, Settings::blastDepth, 0.0f, Settings::fireEvaporate);
					++count;
				}
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
			g_fires.clear();     // a loaded game's refs: scan again at once (hazard handles from the old game; Carbon review 2026-11-17)
			g_fireScanIn = 0;
			g_hazards.clear();
			g_hazardScanIn = 0;
			g_simTime = 0.0f;
			g_windIn.init = false;
			g_windIn.havePrev = false;
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
		// The field's time follows real time down to 15 frames a second (it stopped at 30: at 19 the fog ran at
		// 0.63 of its speed). A frame longer than 1/30 s is taken in two steps (WindRules::StepsPerFrame).
		dt = std::clamp(dt, 0.0f, WindRules::kLongestFrame);
		if (auto* ui = RE::UI::GetSingleton(); ui && ui->GameIsPaused()) {
			return true;  // paused: keep the field as it is (no step, blast clocks stop)
		}
		const int   steps = WindRules::StepsPerFrame(dt);
		const float stepDt = dt / static_cast<float>(steps);
		const float timeBefore = g_simTime;
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
			auto& w = g_windIn;
			if (w.havePrev) {
				const float dv = std::hypot(windX - w.prevX, windY - w.prevY);
				const float ds = std::fabs(speed - w.prevSpeed);
				++w.frames;
				w.sumDv += dv;
				w.maxDv = std::max(w.maxDv, dv);
				w.sumDs += ds;
				w.maxDs = std::max(w.maxDs, ds);
				w.sumV += std::hypot(windX, windY);
			}
			w.prevX = windX;
			w.prevY = windY;
			w.prevSpeed = speed;
			w.havePrev = true;
			// Exponential follow with time constant InputSmoothing; 0 passes the game's value and keeps the follower on
			// it, so switching the setting on in game starts from the current wind without a jump.
			const float tau = Settings::windInputSmoothing;
			const float lastX = w.smoothX;
			const float lastY = w.smoothY;
			if (!w.init || !(tau > 0.0f)) {
				w.smoothX = windX;
				w.smoothY = windY;
				w.init = true;
			} else {
				const float a = 1.0f - std::exp(-dt / tau);
				w.smoothX += (windX - w.smoothX) * a;
				w.smoothY += (windY - w.smoothY) * a;
			}
			if (w.frames > 0) {
				w.sumSmoothDv += std::hypot(w.smoothX - lastX, w.smoothY - lastY);
			}
			windX = w.smoothX;
			windY = w.smoothY;
		}
		const float gustBeforeX = g_gustX;
		const float gustBeforeY = g_gustY;
		g_gustX += windX * dt;
		g_gustY += windY * dt;
		const float baseSpeed = std::hypot(windX, windY);
		g_driftSpeed = baseSpeed;

		WindShaders::SimCB cb{};
		cb.grid[0] = grid.originX;
		cb.grid[1] = grid.originY;
		cb.grid[2] = simCell;
		cb.grid[3] = static_cast<float>(Settings::simResolution);
		cb.time[0] = stepDt;   // time[1..3] are set per step below
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
		// Each body's radius and push are its own row (WindRules::SetDrag).
		cb.dragParams[0] = Settings::actorPushWalk;
		cb.dragParams[1] = Settings::actorFollow;
		cb.dragParams[3] = Settings::actorPushMinSpeed;
		// The wake's dose is the frame's, spread over its steps (WindShaders.h WakeDose); 0 = off.
		cb.dragParams[2] = Settings::wakeEvaporate / static_cast<float>(steps);
		cb.shift[2] = dt;   // the frame's field time: the air's own travel, which the wake's path leaves out (WakeDose)

		RE::NiPoint3 centre{ grid.originX + extent * 0.5f, grid.originY + extent * 0.5f, 0.0f };
		FillActors(cb, centre, now);
		FillBlasts(cb, centre);

		// The actors and impacts are the frame's (held over its steps); the clock and the wind travel advance with
		// each step and end at the values BuildColumns passes (the equilibrium pattern must be the same in both).
		for (int s = 1; s <= steps; ++s) {
			const float part = static_cast<float>(s) / static_cast<float>(steps);
			cb.time[1] = s == steps ? g_simTime : timeBefore + dt * part;
			cb.time[2] = s == steps ? g_gustX : gustBeforeX + windX * dt * part;
			cb.time[3] = s == steps ? g_gustY : gustBeforeY + windY * dt * part;
			g_sim.Step(a_ctx, cb, Settings::enableWalls ? a_heights : nullptr, Settings::pressureIterations);
		}
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
		if (Settings::profiler) {
			UpdateProbe(a_ctx);   // after the build: the relief is this frame's, on this frame's grid
		} else if (g_probe.size != 0) {
			ReleaseProbe();       // the profiler was switched off in the ini
		}
	}

	ID3D11ShaderResourceView* ColumnSrv() { return g_sim.Ready() ? g_sim.ColumnSrv() : nullptr; }
	ID3D11ShaderResourceView* UpperSrv() { return g_sim.Ready() ? g_sim.UpperSrv() : nullptr; }
	ID3D11ShaderResourceView* ReliefSrv() { return g_sim.Ready() ? g_sim.ReliefSrv() : nullptr; }
	bool                      Ready() { return g_sim.Ready() && g_haveGrid && Settings::enableWind; }
	float                     OriginX() { return g_originX; }
	float                     OriginY() { return g_originY; }
	float                     Extent() { return g_extent; }
	float                     GustX() { return g_gustX; }
	float                     SimTime() { return g_simTime; }
	float                     GustY() { return g_gustY; }
	float                     DriftSpeed() { return g_driftSpeed; }

	std::string DescribeInput()
	{
		auto&       w = g_windIn;
		std::string s;
		if (w.frames == 0) {
			s = std::format("wind input: no unpaused frames since the last line (InputSmoothing {:.2f} s)", Settings::windInputSmoothing);
		} else {
			const float n = static_cast<float>(w.frames);
			const float meanV = w.sumV / n;
			s = std::format("wind input: {} frames, game wind {:.1f} units/s, change per frame mean {:.3f} max {:.3f} units/s ({:.2f} % of the wind), "
							"Sky::windSpeed change mean {:.5f} max {:.5f}; field got mean {:.3f} units/s per frame (InputSmoothing {:.2f} s)",
				w.frames, meanV, w.sumDv / n, w.maxDv, meanV > 1e-6f ? 100.0f * w.sumDv / n / meanV : 0.0f, w.sumDs / n, w.maxDs,
				w.sumSmoothDv / n, Settings::windInputSmoothing);
		}
		w.frames = 0;
		w.sumDv = w.maxDv = w.sumDs = w.maxDs = w.sumV = w.sumSmoothDv = 0.0f;
		return s;
	}
	const Stats&              GetStats() { return g_stats; }
}
