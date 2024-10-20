#ifndef DISTRIBUTED_POPCORN_CLUSTER_ASSIGNMENT_H
#define DISTRIBUTED_POPCORN_CLUSTER_ASSIGNMENT_H

// Library imports
#include <mpi.h>

// Local imports
#include "mat/dense_mat.hh"

namespace popcorn {

class ClusterAssignment {
public:
  /**
   * Generates a round robin cluster assignment.
   */
  static ClusterAssignment round_robin(int points, int clusters);

  /**
   * Generates a cluster assignment based on minimum distance.
   *
   * @param D is the dense distance distance matrix.
   */
  static ClusterAssignment min_distance(DenseMat &D);

  /**
   * Returns the total number of points in the assignment.
   *
   * @return the total number of points.
   */
  int64_t get_total_points();

  /**
   * Returns the total number of clusters in the assignment.
   *
   * @return the total number of clusters.
   */
  int64_t get_total_clusters();

  /**
   * Returns the list of point indices.
   *
   * @return the list of point indices.
   */
  std::vector<float> get_points();

  /**
   * Returns the list of cluster indices.
   *
   * @return the list of cluster indices.
   */
  std::vector<float> get_clusters();

  /**
   * Returns the list of points per cluster.
   */
  std::vector<float> get_points_per_cluster();

private:
  // Constructor for ClusterAssignment.
  ClusterAssignment(int64_t total_points, int64_t total_clusters);

  // The list of point indices
  std::vector<float> points;

  // The list of cluster indices
  std::vector<float> clusters;

  // The list of points per cluster
  std::vector<float> points_per_cluster;

  // Total number of points
  int64_t total_points;

  // Total number of clusters
  int64_t total_clusters;

  /**
   * Assigns a point to a cluster.
   *
   * @param point is the point to assign.
   * @param cluster is the cluster to assign.
   */
  void assign(float point, float cluster);
};

} // namespace popcorn

#endif // DISTRIBUTED_POPCORN_CLUSTER_ASSIGNMENT_H
