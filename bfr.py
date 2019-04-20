from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from functools import reduce
from collections import OrderedDict
import numpy as np
import math
import time
import sys

# INPUT_FILE = "hw5_clustering.txt"
# INPUT_CLUSTER = 10
# OUTPUT_FILE = "hw5.txt"

INPUT_FILE = sys.argv[1]
INPUT_CLUSTER = int(sys.argv[2])
OUTPUT_FILE = sys.argv[3]

class BFR:
    def __init__(self):
        self.DS = dict()  # key: id, value: stats
        self.CS = dict()  # key: id, value: stats
        self.RS = list()  # list of point_id

    def init_ds_cs_rs(self, input_data):

        # Step 2:
        initial_data = np.array(input_data)
        n_clusters = 10 * INPUT_CLUSTER
        initial_KMeans = KMeans(n_clusters=n_clusters, n_jobs=-1, max_iter=100).fit(initial_data)

        # Step 3: In the K-Means result from Step 2, move all the clusters with only one point to RS (outliers).
        for i in range(initial_KMeans.n_clusters):
            cluster_item_id = np.where(initial_KMeans.labels_ == i)[0]
            cluster_item_points = initial_data[cluster_item_id].tolist()
            # Move all the clusters with only one point to RS (outliers). (May increase to 10)
            if len(cluster_item_id) <= 10:
                for point in cluster_item_points:
                    # self.RS.append(tuple(point))
                    self.RS.append(data_id_dict[tuple(point)])  # Get the point id and add to RS
                    input_data.remove(point)

        # Step 4: Run K-Means again to cluster the rest of the data point
        #         with K = the number of input clusters.
        # Step 5: Generate the DS clusters from the above K-Means
        self._gen_initial_ds(input_data)

        # Step 6. Run K-Means on the points in the RS with a large K to
        # generate CS (clusters with more than one points)
        # and RS (clusters with only one point).
        initial_cs_data = np.array(list(self.RS))
        large_k = INPUT_CLUSTER * 10
        if len(self.RS) >= large_k:
            self._gen_initial_cs(initial_cs_data, large_k)

    def main(self, other_data):
        # Step 7: Load other 20% data
        # Step 8 - 10:
        self.assign(other_data)

        # Step 11: Run K-Means on the RS with a large K
        # Generate CS (clusters with more than one points) and RS (clusters with only one point).
        large_k = int(len(self.RS) * 0.6)
        if len(self.RS) > 10:
            self.rs_kmeans_gen_new_cs_rs(self.RS, large_k)

        # Step 12: Merge CS clusters that have a Mahalanobis Distance < 2√𝑑.
        if len(self.CS) > 1:
            self.merge_cs(self.CS)

    def _gen_initial_ds(self, input_data):
        # Step 4 & 5:
        remaining_data_init = np.array(input_data)
        initial_ds_kmeans = KMeans(n_clusters=INPUT_CLUSTER, n_jobs=-1, max_iter=30).fit(remaining_data_init)

        for i in range(initial_ds_kmeans.n_clusters):
            cluster_item_id = np.where(initial_ds_kmeans.labels_ == i)[0]
            cluster_item_points = remaining_data_init[cluster_item_id].tolist()
            point_id = []
            for point in cluster_item_points:
                point_id.append(data_id_dict[tuple(point)])
            # ds_cluster_point_list = list(map(lambda x: tuple(x), cluster_item_points))
            # ds_cluster_points_idx
            # ds_cluster_point_list = []

            # the number of points
            N = len(cluster_item_points)
            # vector SUM
            SUM = self.get_SUM(cluster_item_points)
            # vector SUMSQ
            SUMSQ = self.get_SUMSQ(cluster_item_points)

            centroid = self.get_centroid(N, SUM)
            std = self.get_std(N, SUM, SUMSQ)
            self.DS[i] = {"id"  : point_id,
                          "N"       : N,
                          "SUM"     : SUM,
                          "SUMSQ"   : SUMSQ,
                          "centroid": centroid,
                          "std"     : std}
            # DS_stats = (tuple(ds_cluster_point_idx_list), N, tuple(SUM), tuple(SUMSQ))
            # self.DS[i] = DS_stats

    def _gen_initial_cs(self, initial_cs_data, large_k):
        """
        # Step 6
        Run K-Means on the points in the RS with a large K
        Generate initial CS will also update RS
        (remove points that already clustered from RS and add clusters that only have one point to RS)
        :param initial_cs_data: initial rs
        :return:
        """
        initial_cs_kmeans = KMeans(n_clusters=large_k, n_jobs=-1, max_iter=100).fit(initial_cs_data)

        for i in range(initial_cs_kmeans.n_clusters):
            cluster_item_id = np.where(initial_cs_kmeans.labels_ == i)[0]
            cluster_item_points = initial_cs_data[cluster_item_id].tolist()

            # Move all the clusters with only one point to RS

            if len(cluster_item_id) > 2:
                # cs_cluster_points_idx
                point_id = []
                for point in cluster_item_points:
                    point_id.append(data_id_dict[tuple(point)])  # get the point idx and add to list
                    self.RS.remove(data_id_dict[tuple(point)])  # get the point idx and remove from RS

                # the number of points
                N = len(cluster_item_id)
                # vector SUM
                SUM = self.get_SUM(cluster_item_points)
                # vector SUMSQ
                SUMSQ = self.get_SUMSQ(cluster_item_points)

                centroid = self.get_centroid(N, SUM)
                std = self.get_std(N, SUM, SUMSQ)
                self.CS[i] = {"id"      : point_id,
                              "N"       : N,
                              "SUM"     : SUM,
                              "SUMSQ"   : SUMSQ,
                              "centroid": centroid,
                              "std"     : std}

    def get_SUM(self, points):
        """
        :param points: [ [x1,y1,z1,...], [x2,y2,z2,...], [x3,y3,z3,...], ... ]
        :return: sum_: [sum(x1,x2,x3, ...), sum(y1,y2,y3, ...), sum(z1,z2,z3, ...), ...]
        """
        sum_ = reduce(lambda x, y: self.cal_sum(x, y), points)
        return sum_

    def get_SUMSQ(self, points):
        sq = list(map(lambda x: self.squ(x), points))
        sumsq = reduce(lambda x, y: self.cal_sum(x, y), sq)
        return sumsq

    def print_ds_cs_rs_stats(self):
        num_ds_points = 0
        for cluster in list(self.DS.values()):
            num_ds_points += cluster['N']
        print("DS length:", len(self.DS.values()), " Number of DS points: ", num_ds_points)

        num_cs_points = 0
        for cluster in list(self.CS.values()):
            num_cs_points += cluster['N']
        print("CS length:", len(self.CS.values()), " Number of CS points: ", num_cs_points)

        print("RS length:", len(self.RS))

        return num_ds_points, num_cs_points

    def assign(self, other_data):
        """
        # Step 8. For the new points, compare them to each of the DS using the Mahalanobis Distance
        # Assign them to the nearest DS clusters if the distance is < 2√𝑑 (d is the number of dimensions).

        # Step 9: For the new points that are not assigned to DS clusters,
        # using the Mahalanobis Distance and assign the points to the nearest CS clusters if the distance is < 2√𝑑

        # Step 10: For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.

        :param other_data:
        :param ds_centroid_std_dict: ds_cluster_stat: (centroid, std)
        :param cs_centroid_std_dict: cs_cluster_stat: (centroid, std)
        :return:
        """
        for each_data_point in other_data:
            # each_data_point is [x,y,z, ...]
            # Check DS
            # DS = copy.deepcopy(self.DS)
            point, cluster_id = self.get_nearest_point_and_cluster(each_data_point, self.DS)

            if cluster_id is not None:
                self.DS[cluster_id] = self.update_cur_stat(self.DS[cluster_id], each_data_point)
            # Check CS if this point cannot assign to DS
            else:
                if len(self.CS) != 0:
                    point, cluster_id = self.get_nearest_point_and_cluster(each_data_point, self.CS)
                    if point and cluster_id is not None:
                        self.CS[cluster_id] = self.update_cur_stat(self.CS[cluster_id], each_data_point)
                    else:
                        self.RS.append(data_id_dict[tuple(each_data_point)])
                else:
                    self.RS.append(data_id_dict[tuple(each_data_point)])

    def get_nearest_point_and_cluster(self, point, cluster_dict):
        min_distance = float("inf")
        min_point = None
        min_cid = None
        for cid, cluster_stat_dict in cluster_dict.items():
            distance = self.mahalanobis_distance(point, cluster_stat_dict['centroid'], cluster_stat_dict['std'])
            if distance < 2 * math.sqrt(num_dimensions):
                if distance < min_distance:
                    min_distance = distance
                    min_point = point
                    min_cid = cid
        return min_point, min_cid

    def rs_kmeans_gen_new_cs_rs(self, rs, large_k):
        # Step 11:
        # Change point_id in RS to point
        rs_data = []
        for point_id in rs:
            rs_data.append(id_data_dict[point_id])
        rs_data = np.array(rs_data)
        rs_km = KMeans(n_clusters=large_k, n_jobs=-1, max_iter=100).fit(rs_data)
        # cur_cs_length = len(self.CS)
        cid = 0
        for i in range(rs_km.n_clusters):
            # cluster item ids are not same with the origin data id
            cluster_item_id = np.where(rs_km.labels_ == i)[0]
            cluster_item_points = rs_data[cluster_item_id].tolist()
            # Move all the clusters with only one point to RS
            # AKA: If the cluster has more than one point, delete them in the RS
            if len(cluster_item_points) > 1:
                # get cluster item idx
                point_id_list = []
                for point in cluster_item_points:
                    self.RS.remove(data_id_dict[tuple(point)])  # remove in RS
                    point_id_list.append(data_id_dict[tuple(point)])

                N = len(cluster_item_id)  # the number of points
                SUM = self.get_SUM(cluster_item_points)  # vector SUM
                SUMSQ = self.get_SUMSQ(cluster_item_points)  # vector SUMSQ
                centroid = self.get_centroid(N, SUM)
                std = self.get_std(N, SUM, SUMSQ)
                self.CS[cid] = {"id": point_id_list,
                              "N": N,
                              "SUM": SUM,
                              "SUMSQ": SUMSQ,
                              "centroid": centroid,
                              "std": std}
                cid += 1

    def merge_cs(self, cs):

        # cs_stat_list = list(cs.items())  # cs_stat queue == (cid, c_stats_dict)
        # new_cs = dict()

        merge_list = self.find_merge_list(cs)
        if len(merge_list) != 0:
            for two_set in merge_list:
                if two_set[0] != two_set[1] and two_set[0] in cs.keys() and two_set[1] in cs.keys():
                    stat1 = cs[two_set[0]]
                    stat2 = cs[two_set[1]]
                    index = max(cs.keys()) + 1
                    cs[index] = self.merge_two_stat(stat1, stat2)
                    del cs[two_set[0]]
                    del cs[two_set[1]]

        # while cs:
        #     cs_stat = cs.popitem()  # cs_stat[0] == cid, cs_stat[1] == cs_stat_dict
        #     cur_centroid = cs_stat[1]['centroid']
        #     remove_list = []  # (cid, cs_stat_dict)
        #     for next_cs_key, next_cs_stat in cs.items():
        #         distance = self.mahalanobis_distance(cur_centroid, next_cs_stat['centroid'], next_cs_stat['std'])
        #         if distance < 2 * math.sqrt(num_dimensions):
        #             remove_list.append((next_cs_key, next_cs_stat))  # Mark down item to remove
        #             cs.pop(next_cs_key)  # Remove from origin
        #     if len(remove_list) > 0:
        #         # remove item from cs_stat_list & Merge
        #         while remove_list:
        #             remove_item = remove_list.pop(0)
        #             new_cs_stat_dict = self.merge_two_stat(cs_stat[1], remove_item[1])
        #             if len(remove_list) == 0:
        #                 new_cs[cs_stat[0]] = new_cs_stat_dict
        #                 # del self.CS[remove_item[0]]
        #                 break
        #             remove_list.append(new_cs_stat_dict)
        #             # del self.CS[remove_item[0]]

        # self.CS = new_cs

    def find_merge_list(self, cs):
        merge_list = []
        cluster_list = list(cs.keys())
        for i in range(len(cluster_list)):
            for j in range(len(cluster_list)):
                if i < j:
                    centroid = cs[cluster_list[i]]['centroid']
                    temp = dict()
                    temp[0] = cs[cluster_list[j]]
                    point_n_cluster = self.get_nearest_point_and_cluster(centroid, temp)
                    if point_n_cluster is not None:
                        merge_list.append((cluster_list[i], cluster_list[j]))
        return merge_list

    def merge_two_stat(self, cur_stat, next_stat):
        """
        :param cur_stat: cur_stat_dict
        :param next_stat: next_stat_dict
        :return: new_stat: new_stat_dict
        """
        new_id_list = list(set(cur_stat['id']).union(set(next_stat['id'])))
        new_n = len(new_id_list)
        new_point_list = []
        for new_id in new_id_list:
            new_point_list.append(id_data_dict[new_id])
        new_sum = self.get_SUM(new_point_list)
        new_sumsq = self.get_SUMSQ(new_point_list)
        new_centroid = self.get_centroid(new_n, new_sum)
        new_std = self.get_std(new_n, new_sum, new_sumsq)
        res = {'id': new_id_list, 'N': new_n, 'SUM': new_sum, 'SUMSQ': new_sumsq, 'centroid': new_centroid,
               'std': new_std}
        return res

    def update_cur_stat(self, cur_stat, new_point):
        # print(cur_stat)
        id_list = cur_stat['id']
        n = cur_stat['N']
        sum_ = cur_stat['SUM']
        sumsq = cur_stat['SUMSQ']

        new_id_list = id_list
        new_id_list.append(data_id_dict[tuple(new_point)])

        # new_n = len(new_id_list)
        new_n = n + 1
        new_sum = self.update_sum(new_point, sum_)
        new_sumsq = self.update_sumsq(new_point, sumsq)

        centroid = self.get_centroid(new_n, new_sum)
        std = self.get_std(new_n, new_sum, new_sumsq)

        new_stat = {"id": new_id_list,
                    "N": new_n,
                    "SUM": new_sum,
                    "SUMSQ": new_sumsq,
                    "centroid": centroid,
                    "std": std}
        # new_stat = (tuple(new_id_list), new_n, tuple(new_sum), tuple(new_sumsq))
        return new_stat

    @staticmethod
    def cal_sum(x, y):
        res = []
        for i in range(len(x)):
            ith_sum = float(x[i]) + float(y[i])
            res.append(ith_sum)
        return res

    @staticmethod
    def cal_sumsq(x, y):
        res = []
        for i in range(len(x)):
            ith_sum = float(x[i]) + float(y[i])
            res.append(ith_sum)
        return res

    @staticmethod
    def squ(x):
        """
        return squre of each element in list
        :param x: list [x, y, z, ...] a point
        :return: res: list
        """
        res = []
        for i in x:
            ith_square = float(i) ** 2
            res.append(ith_square)
        return res

    @staticmethod
    def get_centroid(num_points, vector_sum):
        centroid = list(map(lambda x: x / num_points, vector_sum))
        return centroid

    @staticmethod
    def get_std(num_points, vector_sum, vector_sumsq):
        # get variance list
        variance_list = []
        for i in range(len(vector_sumsq)):
            ith_var = (vector_sumsq[i] / num_points) - (vector_sum[i] / num_points) ** 2
            variance_list.append(ith_var)

        # get standard deviation list
        std_list = list(map(lambda x: math.sqrt(x), variance_list))
        # dimension_std[cluster_id] = std_list
        return std_list

    @staticmethod
    def mahalanobis_distance(point, centroid, std):
        sum_distance = 0
        for i in range(num_dimensions):
            sum_distance += ((float(point[i]) - centroid[i]) / std[i]) ** 2
        res = math.sqrt(sum_distance)
        return res

    @staticmethod
    def update_sum(new_point, sum_):
        res = []
        for d in range(num_dimensions):
            new_sum = sum_[d] + float(new_point[d])
            res.append(new_sum)
        return res

    @staticmethod
    def update_sumsq(new_point, sumsq):
        new_sumsq = []
        for d in range(num_dimensions):
            ith_new_sumsq = sumsq[d] + float(new_point[d]) ** 2
            new_sumsq.append(ith_new_sumsq)
        return new_sumsq


def load_data(in_file):
    """
    :param in_file: INPUT_FILE
    :return: feature_list, point_id_dict, number_of_clusters
    """
    f = open(in_file)
    input_data = f.readlines()

    sample = list(map(lambda x: x.strip("\n").split(","), input_data))

    cluster_list = []
    feature_list = []
    data_id = {}  # point: id ==> tuple: int
    point_cluster = {}  # point_id: cid  Ground Truth dict

    for line in sample:
        point_id, cluster_id, feature = line[0], line[1], line[2:]

        data_id[tuple(feature)] = int(point_id)
        cluster_list.append(cluster_id)
        feature_list.append(feature)
        point_cluster[point_id] = cluster_id

    num_of_clusters = len(set(cluster_list))

    return feature_list, data_id, num_of_clusters, point_cluster


if __name__ == '__main__':

    # Step 1:
    data, data_id_dict, num_clusters, ground_truth = load_data(INPUT_FILE)
    id_data_dict = dict((v, k) for k, v in data_id_dict.items())

    n_sample = len(data)  # the number of data points we have
    percentage = 0.2  # percentage of the data points to be load in the memory
    init_data = data[:int(n_sample * percentage)]  # generate the data points for initialization

    num_dimensions = len(init_data[0])
    bfr = BFR()

    # Step 2 - 6:
    start_time = time.time()
    bfr.init_ds_cs_rs(init_data)

    print("Round 1: ")
    num_ds_points, num_cs_points = bfr.print_ds_cs_rs_stats()

    f = open(OUTPUT_FILE, 'w')
    f.write("The intermediate results:\n")
    f.write('Round 1: ' + str(num_ds_points) + "," + str(len(bfr.CS)) + "," + str(num_cs_points) + "," + str(len(
        bfr.RS)) + "\n")

    start = int(n_sample * percentage)
    end = start + int(n_sample * percentage)
    count = 2

    while start < n_sample:
        # Repeat Step 7 - 12
        print("Round: ", count)
        bfr.main(data[start:end])
        num_ds_points, num_cs_points = bfr.print_ds_cs_rs_stats()
        f.write('Round ' + str(count) + ": " + str(num_ds_points) + "," + str(len(bfr.CS)) + "," + str(num_cs_points)
                + "," + str(len(bfr.RS)) + "\n")

        start = end
        end = start + int(n_sample * percentage)
        count += 1
        if n_sample - start <= int(n_sample * percentage):
            bfr.main(data[start:])
            # assign CS to DS if last round
            for cs_id, cs_stat in bfr.CS.items():
                cs_centroid = cs_stat['centroid']
                min_point, ds_cluster_id = bfr.get_nearest_point_and_cluster(cs_centroid, bfr.DS)
                if min_point and ds_cluster_id is not None:  # merge CS to the closest DS
                    bfr.DS[ds_cluster_id] = bfr.merge_two_stat(cs_stat, bfr.DS[ds_cluster_id])

            print("Round: ", count)
            num_ds_points, num_cs_points = bfr.print_ds_cs_rs_stats()
            f.write('Round ' + str(count) + ": " + str(num_ds_points) + "," + str(len(bfr.CS)) + "," + str(num_cs_points) + "," + str(len(bfr.RS)) + "\n")
            break

    end_time = time.time()
    print("Total time: ", end_time - start_time)

    # Actual Cluster
    pre = {}
    for c_id, cluster in bfr.DS.items():
        p_id_list = cluster['id']
        for p_id in p_id_list:
            pre[p_id] = c_id

    # Outliers
    for rs_id in bfr.RS:
        pre[rs_id] = -1
    for c_id, cluster in bfr.CS.items():
        c_id_list = cluster['id']
        for p_id in c_id_list:
            pre[p_id] = -1

    result = OrderedDict(sorted(pre.items()))

    f.write("\n")
    f.write("The clustering results:\n")
    for key, value in result.items():
        f.write(str(key) + ", " + str(value) + "\n")
    f.close()
    acc = normalized_mutual_info_score(list(ground_truth.values()), list(result.values()))

    print(acc)
