#pragma once
#include <glog\logging.h>
#include "sampler.h"
#include "random.h"


class RandomSampler : public Sampler {
public:
	explicit RandomSampler(const size_t num_samples);

	void Initialize(const size_t total_num_samples) override;

	size_t MaxNumSamples() override;

	std::vector<size_t> Sample() override;

private:
	const size_t num_samples_;
	std::vector<size_t> sample_idxs_;
};

