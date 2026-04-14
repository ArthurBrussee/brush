// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.
const OFFSET: u32 = 42;
const WG: u32 = 256;

const BITS_PER_PASS: u32 = 4;
const BIN_COUNT: u32 = 1u << BITS_PER_PASS;
const HISTOGRAM_SIZE: u32 = WG * BIN_COUNT;
const ELEMENTS_PER_THREAD: u32 = 4;

const BLOCK_SIZE = WG * ELEMENTS_PER_THREAD;

// Upper bound on the number of subgroups inside a workgroup of size WG.
// Subgroup size varies by hardware: 8/16 on some Intel, 32 on Apple/most Intel/
// NVIDIA, 64 on AMD wave64. With WG=256 the worst case is SG=8, which gives
// 32 subgroups. We pad `partials` arrays to 32 so they are correctly sized
// for any subgroup size in [8, 64].
//
// Cross-subgroup combine strategy used by every workgroup-wide scan helper:
//
//   if num_subgroups <= subgroup_size:
//       // Fast path: a single subgroupExclusiveAdd in subgroup 0 scans all
//       // per-subgroup totals in parallel. Hits on every common GPU
//       // (SG=32, SG=64) because WG=256 gives at most 8 subgroups.
//   else:
//       // Slow path: thread 0 serializes the combine. Used only when
//       // num_subgroups exceeds the subgroup size (SG=8 with WG=256, where
//       // num_subgroups=32). Rare in practice, but required for correctness.
//
// Both branches are gated on a workgroup-uniform condition so the subgroup
// op stays in uniform control flow.
const MAX_SUBGROUPS: u32 = 32u;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

// Compute linear workgroup ID from 2D dispatch
fn get_workgroup_id(wid: vec3u, num_wgs: vec3u) -> u32 {
    return wid.x + wid.y * num_wgs.x;
}
