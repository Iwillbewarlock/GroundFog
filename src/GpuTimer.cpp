#include "PCH.h"

#include "GpuTimer.h"

bool GpuTimer::Create(ID3D11Device* a_device)
{
	D3D11_QUERY_DESC disjoint{ D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };
	D3D11_QUERY_DESC stamp{ D3D11_QUERY_TIMESTAMP, 0 };
	for (auto& slot : _slots) {
		if (FAILED(a_device->CreateQuery(&disjoint, &slot.disjoint)) ||
			FAILED(a_device->CreateQuery(&stamp, &slot.begin)) ||
			FAILED(a_device->CreateQuery(&stamp, &slot.end))) {
			logger::error("GpuTimer: CreateQuery failed; GPU times will not be measured");
			Release();
			return false;
		}
	}
	_ready = true;
	return true;
}

void GpuTimer::Release()
{
	for (auto& slot : _slots) {
		for (auto** q : { &slot.disjoint, &slot.begin, &slot.end }) {
			if (*q) {
				(*q)->Release();
				*q = nullptr;
			}
		}
		slot.pending = false;
	}
	_ready = false;
}

void GpuTimer::Collect(ID3D11DeviceContext* a_ctx, Slot& a_slot)
{
	if (!a_slot.pending) {
		return;
	}
	D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj{};
	UINT64                              t0 = 0;
	UINT64                              t1 = 0;
	constexpr UINT                      kNoFlush = D3D11_ASYNC_GETDATA_DONOTFLUSH;
	if (a_ctx->GetData(a_slot.disjoint, &dj, sizeof(dj), kNoFlush) != S_OK ||
		a_ctx->GetData(a_slot.begin, &t0, sizeof(t0), kNoFlush) != S_OK ||
		a_ctx->GetData(a_slot.end, &t1, sizeof(t1), kNoFlush) != S_OK) {
		return;  // not ready yet; the slot is overwritten when its turn comes again
	}
	a_slot.pending = false;
	if (dj.Disjoint || dj.Frequency == 0 || t1 < t0) {
		return;
	}
	const double ms = static_cast<double>(t1 - t0) * 1000.0 / static_cast<double>(dj.Frequency);
	++_samples;
	_totalMs += ms;
	if (ms > _maxMs) {
		_maxMs = ms;
		_maxSample = _samples;
	}
}

void GpuTimer::Begin(ID3D11DeviceContext* a_ctx)
{
	if (!_ready) {
		return;
	}
	_current = (_current + 1) % kRing;
	auto& slot = _slots[_current];
	Collect(a_ctx, slot);
	slot.pending = false;
	a_ctx->Begin(slot.disjoint);
	a_ctx->End(slot.begin);
}

void GpuTimer::End(ID3D11DeviceContext* a_ctx)
{
	if (!_ready) {
		return;
	}
	auto& slot = _slots[_current];
	a_ctx->End(slot.end);
	a_ctx->End(slot.disjoint);
	slot.pending = true;
	for (std::size_t i = 1; i < kRing; ++i) {
		Collect(a_ctx, _slots[(_current + i) % kRing]);
	}
}

void GpuTimer::ResetStats()
{
	_samples = 0;
	_totalMs = 0.0;
	_maxMs = 0.0;
	_maxSample = 0;
}
