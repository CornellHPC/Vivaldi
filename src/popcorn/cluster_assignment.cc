// C++ standard imports
#include <cassert>

// Local imports
#include "cluster_assignment.hh"

namespace popcorn {

ClusterAssignment ClusterAssignment::round_robin(int points, int clusters) {
  ClusterAssignment assignment = ClusterAssignment(points, clusters);
  for (int i = 0; i < points; ++i) {
    assignment.assign(i, i % clusters);
  }
  return assignment;
}

ClusterAssignment ClusterAssignment::min_distance(DenseMat &D) {
  // TODO: Implement
  return ClusterAssignment(0, 0);
}

int64_t ClusterAssignment::get_total_points() { return total_points; }

int64_t ClusterAssignment::get_total_clusters() { return total_clusters; }

std::vector<float> ClusterAssignment::get_points() { return points; }

std::vector<float> ClusterAssignment::get_clusters() { return clusters; }

std::vector<float> ClusterAssignment::get_points_per_cluster() {
  return points_per_cluster;
}

ClusterAssignment::ClusterAssignment(int64_t total_points,
                                     int64_t total_clusters) {
  this->total_points = total_points;
  this->total_clusters = total_clusters;
  this->points_per_cluster.resize(total_clusters, 0);
}

void ClusterAssignment::assign(float point, float cluster) {
  assert((cluster >= 0 && cluster < total_clusters) &&
         "Must assign point to a valid cluster!");
  assert(points.size() < total_points && "All points have been assigned!");

  points.push_back(point);
  clusters.push_back(cluster);
  points_per_cluster[cluster] += 1;
}

} // namespace popcorn
