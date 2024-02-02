import json
import numpy as np

HISTOGRAM_LEN = 1024

class HistogramHelper():

    def calculate_value_counts_and_percentages(self, arr):
        value_counts = {}
        total_values = len(arr)

        for value in arr:
            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1
        value_counts = {key: value_counts[key] for key in sorted(value_counts.keys())}

        value_percentages = {}
        for value, count in value_counts.items():
            percentage = count / total_values
            value_percentages[value] = percentage

        return value_percentages

    def gen_histogram_from_entities(self, entities):
        histogram = np.zeros([len(entities), HISTOGRAM_LEN + 1], dtype=float)
        for col_i, col in enumerate(entities):
            hist = []
            cell_list = [cell[1][1] for cell in col]
            cell_set = set(cell_list)
            if len(cell_set) > HISTOGRAM_LEN:
                continue
            value_percentages = self.calculate_value_counts_and_percentages(cell_list)
            for i, percent in enumerate(value_percentages.values()):
                hist.append(percent)
            hist = sorted(hist, reverse=True)
            histogram[col_i][0] = len(hist) / HISTOGRAM_LEN
            for i, percent in enumerate(hist):
                histogram[col_i][i+1] = percent
        return histogram

    def reformat_mysql_histogram_buckets(self, one_histogram, histogram_num_sort):
        histogram_data = json.loads(one_histogram)
        buckets_data = histogram_data['buckets']
        mode = histogram_data['histogram-type']
        if mode == 'singleton':
            for bucket_idx in range(len(buckets_data)):
                bucket = buckets_data[bucket_idx]
                encode_str = str(bucket[0]).split(':')[-1]

                if encode_str == '':
                    continue

                if bucket_idx == 0:
                    histogram_num_sort.append(bucket[1])
                else:
                    histogram_num_sort.append(bucket[1] - buckets_data[bucket_idx - 1][1])

            sorted_data = sorted(histogram_num_sort, reverse=True)
            total_probability = sum(item for item in sorted_data)
            # Recalculate probability
            recomputed_data = [round(item / total_probability, 3) for item in sorted_data]
            histogram = recomputed_data
            histogram.insert(0, len(histogram)/HISTOGRAM_LEN)
            histogram = histogram + [0] * (HISTOGRAM_LEN+1 - len(histogram))
        else:
            histogram = [0] * (HISTOGRAM_LEN+1)
        return histogram

    def reformat_mysql_histograms(self, mysql_histogram):
        table_column_name_2_histogram = {}
        for one_histogram_and_table_mse in mysql_histogram:
            table_name_his = one_histogram_and_table_mse[1]
            column_name_his = one_histogram_and_table_mse[2]
            one_histogram = one_histogram_and_table_mse[3]

            key = table_name_his + "||" + column_name_his
            refarmatted_histogram = []
            refarmatted_histogram = self.reformat_mysql_histogram_buckets(one_histogram, refarmatted_histogram)
            table_column_name_2_histogram[key] = refarmatted_histogram
        return table_column_name_2_histogram