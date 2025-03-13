#include "volsurfs/RaySamplesPacked.cuh"

#include "volsurfs/UtilsPytorch.h"

#include "volsurfs/RaySamplesPackedGPU.cuh"

template <typename T>
T div_round_up(T val, T divisor)
{
    return (val + divisor - 1) / divisor;
}

RaySamplesPacked::RaySamplesPacked(const int nr_rays, const int max_nr_samples, const int first_sample_idx, const int values_dim)
{
    // per sample
    samples_idx = torch::arange(first_sample_idx, max_nr_samples + first_sample_idx, torch::dtype(torch::kInt32).device(torch::kCUDA, 0)).unsqueeze(1);
    samples_3d = torch::zeros({max_nr_samples, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    samples_3d.fill_(-1.0);
    samples_dirs = torch::zeros({max_nr_samples, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    samples_dirs.fill_(-1.0);
    samples_z = torch::zeros({max_nr_samples, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    samples_z.fill_(-1.0);
    samples_dt = torch::zeros({max_nr_samples, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    samples_dt.fill_(-1.0);
    
    // per ray
    ray_start_end_idx = torch::zeros({nr_rays, 2}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    ray_start_end_idx.fill_(-1.0);
    ray_o = torch::zeros({nr_rays, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    ray_o.fill_(-1.0);
    ray_d = torch::zeros({nr_rays, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    ray_d.fill_(-1.0);
    ray_enter = torch::zeros({nr_rays, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    ray_enter.fill_(-1.0);
    ray_exit = torch::zeros({nr_rays, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    ray_exit.fill_(-1.0);
    ray_max_dt = torch::zeros({nr_rays, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    ray_max_dt.fill_(-1.0);

    // TODO: remove values_dim from constructor, make it more flexible
    // per sample values (optional)
    samples_values = torch::zeros({max_nr_samples, values_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    samples_values.fill_(-1.0);

    has_samples_values = false;
    has_dt = false;
    is_compacted = true;
}

int RaySamplesPacked::get_nr_rays() const
{
    return this->ray_start_end_idx.size(0);
}

float RaySamplesPacked::get_ray_max_dt(int ray_idx) const
{
    // make sure ray_idx is valid
    if (ray_idx < 0 || ray_idx >= this->get_nr_rays())
    {
        throw std::invalid_argument("ray_idx must be in the range [0, nr_rays)");
    }
    return this->ray_max_dt.index({ray_idx, 0}).item<float>();
}

torch::Tensor RaySamplesPacked::get_ray_samples_idx(int ray_idx) const
{
    torch::Tensor ray_start_idx = this->ray_start_end_idx.index({ray_idx, 0});
    torch::Tensor ray_end_idx = this->ray_start_end_idx.index({ray_idx, 1});
    torch::Tensor ray_samples = this->samples_idx.index({torch::indexing::Slice(ray_start_idx.item<int>(), ray_end_idx.item<int>())});
    return ray_samples;
}

torch::Tensor RaySamplesPacked::get_ray_samples_3d(int ray_idx) const
{
    torch::Tensor ray_start_idx = this->ray_start_end_idx.index({ray_idx, 0});
    torch::Tensor ray_end_idx = this->ray_start_end_idx.index({ray_idx, 1});
    torch::Tensor ray_samples = this->samples_3d.index({torch::indexing::Slice(ray_start_idx.item<int>(), ray_end_idx.item<int>())});
    return ray_samples;
}

torch::Tensor RaySamplesPacked::get_ray_samples_dirs(int ray_idx) const
{
    torch::Tensor ray_start_idx = this->ray_start_end_idx.index({ray_idx, 0});
    torch::Tensor ray_end_idx = this->ray_start_end_idx.index({ray_idx, 1});
    torch::Tensor ray_samples = this->samples_dirs.index({torch::indexing::Slice(ray_start_idx.item<int>(), ray_end_idx.item<int>())});
    return ray_samples;
}

torch::Tensor RaySamplesPacked::get_ray_samples_z(int ray_idx) const
{
    torch::Tensor ray_start_idx = this->ray_start_end_idx.index({ray_idx, 0});
    torch::Tensor ray_end_idx = this->ray_start_end_idx.index({ray_idx, 1});
    torch::Tensor ray_samples = this->samples_z.index({torch::indexing::Slice(ray_start_idx.item<int>(), ray_end_idx.item<int>())});
    return ray_samples;
}

torch::Tensor RaySamplesPacked::get_ray_samples_dt(int ray_idx) const
{
    torch::Tensor ray_start_idx = this->ray_start_end_idx.index({ray_idx, 0});
    torch::Tensor ray_end_idx = this->ray_start_end_idx.index({ray_idx, 1});
    torch::Tensor ray_samples = this->samples_dt.index({torch::indexing::Slice(ray_start_idx.item<int>(), ray_end_idx.item<int>())});
    return ray_samples;
}

torch::Tensor RaySamplesPacked::get_samples_values() const
{
    // CHECK(has_samples_values) << "RaySamplesPacked does not have samples values";
    return samples_values;
}

torch::Tensor RaySamplesPacked::get_ray_samples_values(int ray_idx) const
{
    CHECK(has_samples_values) << "RaySamplesPacked does not have samples values";
    torch::Tensor ray_start_idx = this->ray_start_end_idx.index({ray_idx, 0});
    torch::Tensor ray_end_idx = this->ray_start_end_idx.index({ray_idx, 1});
    torch::Tensor ray_samples = this->samples_values.index({torch::indexing::Slice(ray_start_idx.item<int>(), ray_end_idx.item<int>())});
    return ray_samples;
}

torch::Tensor RaySamplesPacked::get_ray_start_end_idx(int ray_idx) const
{
    torch::Tensor ray_start_end_idx = this->ray_start_end_idx.index({ray_idx, torch::indexing::Slice()});
    return ray_start_end_idx;
}

torch::Tensor RaySamplesPacked::get_ray_o(int ray_idx) const
{
    torch::Tensor ray_o = this->ray_o.index({ray_idx, torch::indexing::Slice()});
    return ray_o;
}

torch::Tensor RaySamplesPacked::get_ray_d(int ray_idx) const
{
    torch::Tensor ray_d = this->ray_d.index({ray_idx, torch::indexing::Slice()});
    return ray_d;
}

torch::Tensor RaySamplesPacked::get_ray_enter(int ray_idx) const
{
    torch::Tensor ray_enter = this->ray_enter.index({ray_idx, torch::indexing::Slice()});
    return ray_enter;
}

torch::Tensor RaySamplesPacked::get_ray_exit(int ray_idx) const
{
    torch::Tensor ray_exit = this->ray_exit.index({ray_idx, torch::indexing::Slice()});
    return ray_exit;
}

int RaySamplesPacked::get_max_nr_samples() const
{
    return this->samples_idx.size(0);
}

bool RaySamplesPacked::is_empty() const
{
    if (get_nr_rays() == 0) return true;
    if (get_total_nr_samples() == 0) return true;
    return false;
}

torch::Tensor RaySamplesPacked::get_nr_samples_per_ray() const
{   
    torch::Tensor ray_start_idx = this->ray_start_end_idx.index({torch::indexing::Slice(), 0});
    torch::Tensor ray_end_idx = this->ray_start_end_idx.index({torch::indexing::Slice(), 1});
    torch::Tensor nr_samples_per_ray = ray_end_idx - ray_start_idx;
    return nr_samples_per_ray;
}

int RaySamplesPacked::get_total_nr_samples() const
{
    torch::Tensor nr_samples_per_ray = this->get_nr_samples_per_ray();
    int nr_samples_total = nr_samples_per_ray.sum().item<int>();
    return nr_samples_total;
}

int RaySamplesPacked::get_values_dim() const
{
    // CHECK(has_samples_values) << "RaySamplesPacked does not have samples values";
    return this->samples_values.size(1);
}

bool RaySamplesPacked::are_samples_values_set() const
{
    return this->has_samples_values;
}

RaySamplesPacked RaySamplesPacked::compact_to_valid_samples()
{
    if (verbose)
        std::cout << "compact_to_valid_samples" << std::endl;

    CHECK(!is_compacted) << "RaySamplesPacked must not be compacted before calling compact_to_valid_samples";

    int nr_rays = this->get_nr_rays();
    int nr_compacted_samples = this->get_total_nr_samples();
    // std::cout << "nr_compacted_samples " << nr_compacted_samples << std::endl;
    int nr_uncompacted_samples = this->get_max_nr_samples();
    // std::cout << "nr_uncompacted_samples " << nr_uncompacted_samples << std::endl;
    int values_dim = this->get_values_dim();
    RaySamplesPacked compact_ray_samples_packed(nr_rays, nr_compacted_samples, 0, values_dim);
    compact_ray_samples_packed.ray_o = this->ray_o.clone();
    compact_ray_samples_packed.ray_d = this->ray_d.clone();
    compact_ray_samples_packed.ray_enter = this->ray_enter.clone();
    compact_ray_samples_packed.ray_exit = this->ray_exit.clone();
    compact_ray_samples_packed.has_samples_values = this->has_samples_values;
    compact_ray_samples_packed.has_dt = this->has_dt;
    compact_ray_samples_packed.ray_max_dt = this->ray_max_dt.clone();
    compact_ray_samples_packed.is_compacted = true;

    if (this->is_empty())
    {
        return compact_ray_samples_packed;
    }

    torch::Tensor ray_start_idx = this->ray_start_end_idx.index({torch::indexing::Slice(), 0});
    torch::Tensor ray_end_idx = this->ray_start_end_idx.index({torch::indexing::Slice(), 1});
    torch::Tensor nr_samples_per_ray = this->get_nr_samples_per_ray();
    torch::Tensor out_indices_start = nr_samples_per_ray.cumsum(0).to(torch::kInt32);
    // drop the last element and add a 0 at the beginning
    out_indices_start = torch::cat({torch::zeros(1, torch::dtype(torch::kInt32).device(torch::kCUDA, 0)), out_indices_start.slice(0, 0, -1)}, 0);

    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    RaySamplesPackedGPU::compact_to_valid_samples_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        nr_uncompacted_samples,
        nr_compacted_samples,
        values_dim,
        // input
        this->samples_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        this->samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        this->samples_dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        this->samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        this->samples_dt.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        this->samples_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        this->ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        //
        out_indices_start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        // output
        compact_ray_samples_packed.samples_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        compact_ray_samples_packed.samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        compact_ray_samples_packed.samples_dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        compact_ray_samples_packed.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        compact_ray_samples_packed.samples_dt.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        compact_ray_samples_packed.samples_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        compact_ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
    );

    // wait for the kernel to finish
    cudaDeviceSynchronize();
    // check if any error occurred, if so print it
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
    else {
        if (verbose)
            std::cout << "No error" << std::endl;
    }

    if (verbose) {
        std::cout << "nr_rays " << compact_ray_samples_packed.get_nr_rays() << std::endl;
        std::cout << "max_nr_samples " << compact_ray_samples_packed.get_max_nr_samples() << std::endl;
        std::cout << "nr_samples " << compact_ray_samples_packed.get_total_nr_samples() << std::endl;
    }

    // assert 
    CHECK(compact_ray_samples_packed.get_total_nr_samples() == compact_ray_samples_packed.get_max_nr_samples()) << "Compact packed must have nr of samples matching its max nr of samples";

    return compact_ray_samples_packed;
}


// RaySamplesPacked RaySamplesPacked::filter_samples_in_range(
//     const torch::Tensor &samples_3d,
//     const torch::Tensor &samples_dirs,
//     const torch::Tensor &t_near,
//     const torch::Tensor &t_far
// ) {
//     if (verbose)
//         std::cout << "filter_samples_in_range" << std::endl;

//     int nr_rays = this->get_nr_rays();
//     int max_nr_samples = this->get_total_nr_samples();
//     int values_dim = this->get_values_dim();
//     RaySamplesPacked ray_samples_packed_filtered(nr_rays, max_nr_samples, 0, values_dim);
//     ray_samples_packed_filtered.has_samples_values = this->has_samples_values;
//     ray_samples_packed_filtered.has_dt = false;
//     ray_samples_packed_filtered.ray_max_dt = this->ray_max_dt;

//     // Define CUDA block configuration
//     const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

//     // Call CUDA kernel to extract instance samples
//     RaySamplesPackedGPU::filter_samples_in_range_gpu<<<blocks, BLOCK_SIZE>>>(
//         nr_rays,
//         this->samples_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
//         samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         samples_dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         this->samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         this->samples_dt.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         this->ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
//         this->ray_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         this->ray_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         this->ray_enter.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         this->ray_exit.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         //
//         t_near.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         t_far.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         // Output tensors
//         ray_samples_packed_filtered.samples_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
//         ray_samples_packed_filtered.samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         ray_samples_packed_filtered.samples_dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         ray_samples_packed_filtered.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         ray_samples_packed_filtered.samples_dt.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         ray_samples_packed_filtered.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
//         ray_samples_packed_filtered.ray_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         ray_samples_packed_filtered.ray_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         ray_samples_packed_filtered.ray_enter.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         ray_samples_packed_filtered.ray_exit.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
//     );

//     // wait for the kernel to finish
//     cudaDeviceSynchronize();
//     // check if any error occurred, if so print it
//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess)
//     {
//         std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
//     }
//     else {
//         if (verbose)
//             std::cout << "No error" << std::endl;
//     }
    
//     ray_samples_packed_filtered = ray_samples_packed_filtered.compact_to_valid_samples();

//     // Return the filtered samples
//     return ray_samples_packed_filtered;
// }

RaySamplesPacked RaySamplesPacked::copy() const
{   
    if (verbose)
        std::cout << "copy" << std::endl;

    RaySamplesPacked ray_samples_packed_copy(this->get_nr_rays(), this->get_max_nr_samples(), 0, this->get_values_dim());

    ray_samples_packed_copy.samples_idx = this->samples_idx.clone();
    ray_samples_packed_copy.samples_3d = this->samples_3d.clone();
    ray_samples_packed_copy.samples_dirs = this->samples_dirs.clone();
    ray_samples_packed_copy.samples_z = this->samples_z.clone();
    ray_samples_packed_copy.samples_dt = this->samples_dt.clone();
    if (has_samples_values)
        ray_samples_packed_copy.samples_values = this->samples_values.clone();
    ray_samples_packed_copy.ray_start_end_idx = this->ray_start_end_idx.clone();
    ray_samples_packed_copy.ray_o = this->ray_o.clone();
    ray_samples_packed_copy.ray_d = this->ray_d.clone();
    ray_samples_packed_copy.ray_enter = this->ray_enter.clone();
    ray_samples_packed_copy.ray_exit = this->ray_exit.clone();
    ray_samples_packed_copy.has_samples_values = this->has_samples_values;
    ray_samples_packed_copy.has_dt = this->has_dt;
    ray_samples_packed_copy.is_compacted = this->is_compacted;
    ray_samples_packed_copy.ray_max_dt = this->ray_max_dt.clone();

    return ray_samples_packed_copy;
}

void RaySamplesPacked::set_samples_values(const torch::Tensor &samples_values)
{
    if (verbose)
        std::cout << "set_samples_values" << std::endl;

    CHECK(is_compacted) << "RaySamplesPacked must be compacted before calling set_samples_values";
    CHECK(!has_samples_values) << "Trying to set samples_values when it is already set, remove them first if you want to replace them";
    CHECK(samples_values.dim() == 2) << "samples_values must be 2-dimensional";
    CHECK(samples_3d.size(0) == samples_values.size(0)) << "samples_3d and samples_values do not have matching 0 dimension";
    
    // copy the values
    this->samples_values = samples_values.clone();
    this->has_samples_values = true;
}
void RaySamplesPacked::remove_samples_values()
{   
    if (verbose)
        std::cout << "remove_samples_values" << std::endl;

    CHECK(has_samples_values) << "trying to remove samples_values when it is not set";
    // set to non-initialized
    this->samples_values.fill_(-1.0);
    has_samples_values = false;
}

void RaySamplesPacked::update_dt(
    const bool is_background
)
{
    if (verbose)
        std::cout << "update_dt" << std::endl;

    CHECK(!is_empty()) << "RaySamplesPacked must not be empty before calling update_dt";
    CHECK(is_compacted) << "RaySamplesPacked must be compacted before calling update_dt";

    int nr_rays = this->get_nr_rays();
    int nr_samples = this->get_total_nr_samples();

    if (verbose) {
        std::cout << "nr_rays " << nr_rays << std::endl;
        std::cout << "is_background " << is_background << std::endl;
        std::cout << "max_nr_samples " << get_max_nr_samples() << std::endl;
        std::cout << "nr_samples " << get_total_nr_samples() << std::endl;
        std::cout << "ray_start_end_idx dims " << ray_start_end_idx.sizes() << std::endl;
        std::cout << "samples_z dims " << samples_z.sizes() << std::endl;
        std::cout << "samples_dt dims " << samples_dt.sizes() << std::endl;
        std::cout << "ray_max_dt dims " << ray_max_dt.sizes() << std::endl;
    }

    
    // TODO: remove
    // set all samples_dt to be 0.001
    // this->samples_dt.fill_(0.001);

    // Define CUDA block configuration
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    // Call CUDA kernel to update dt
    RaySamplesPackedGPU::update_dt_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        nr_samples,
        // input
        is_background,
        this->ray_max_dt.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        this->ray_exit.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        this->samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        this->ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        // output
        this->samples_dt.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    // wait for the kernel to finish
    cudaDeviceSynchronize();
    // check if any error occurred, if so print it
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
    else {
        if (verbose)
            std::cout << "No error" << std::endl;
    }

    // float ray_max_dt = this->ray_max_dt.item<float>();
    // float max_dt = this->samples_dt.max().item<float>();
    // float eps = 1e-5;
    // CHECK(max_dt <= (ray_max_dt + eps)) << "max_dt " << max_dt << " must be less than or equal to ray_max_dt " << ray_max_dt;

    has_dt = true;
}

// void RaySamplesPacked::update_dt_in_grid_occupied_regions(
//     const int nr_voxels_per_dim,
//     const Eigen::Vector3f grid_extent,
//     const torch::Tensor &grid_occupancy,
//     const torch::Tensor &grid_roi
// )
// {
//     if (verbose)
//         std::cout << "update_dt_in_grid_occupied_regions" << std::endl;

//     int nr_rays = this->get_nr_rays();

//     // Define CUDA block configuration
//     const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

//     // Call CUDA kernel to update dt
//     RaySamplesPackedGPU::update_dt_in_grid_occupied_regions_gpu<<<blocks, BLOCK_SIZE>>>(
//         nr_rays,
//         // input
//         this->ray_max_dt.item<float>(),
//         this->ray_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         this->ray_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         this->ray_exit.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         this->samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         this->samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         this->ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
//         // occupancy grid
//         nr_voxels_per_dim, // m_nr_voxels_per_dim
//         grid_extent, // m_grid_extent
//         grid_occupancy.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(), // m_grid_occupancy
//         grid_roi.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(), // m_grid_roi
//         // output
//         this->samples_dt.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
//     );

//     // wait for the kernel to finish
//     cudaDeviceSynchronize();
//     // check if any error occurred, if so print it
//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess)
//     {
//         std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
//     }
//     else {
//         if (verbose)
//             std::cout << "No error" << std::endl;
//     }

//     has_dt = true;
// }