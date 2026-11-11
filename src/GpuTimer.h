#pragma once

// GPU time of the fog pass from D3D11 timestamp queries. Results are read a few frames later
// without stalling (GetData with DONOTFLUSH); a frame whose result is not ready is skipped.
class GpuTimer
{
public:
	bool Create(ID3D11Device* a_device);
	void Release();

	void Begin(ID3D11DeviceContext* a_ctx);
	void End(ID3D11DeviceContext* a_ctx);

	[[nodiscard]] std::uint32_t Samples() const { return _samples; }
	[[nodiscard]] double        AverageMs() const { return _samples ? _totalMs / _samples : 0.0; }
	[[nodiscard]] double        MaxMs() const { return _maxMs; }
	[[nodiscard]] std::uint32_t MaxSample() const { return _maxSample; }   // which sample was the slowest (1 = first)
	void                        ResetStats();

private:
	static constexpr std::size_t kRing = 4;

	struct Slot
	{
		ID3D11Query* disjoint{ nullptr };
		ID3D11Query* begin{ nullptr };
		ID3D11Query* end{ nullptr };
		bool         pending{ false };
	};

	void Collect(ID3D11DeviceContext* a_ctx, Slot& a_slot);

	std::array<Slot, kRing> _slots{};
	std::size_t             _current{ 0 };
	std::uint32_t           _samples{ 0 };
	double                  _totalMs{ 0.0 };
	double                  _maxMs{ 0.0 };
	std::uint32_t           _maxSample{ 0 };
	bool                    _ready{ false };
};
