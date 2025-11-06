import matplotlib.pyplot as plt
import argparse
from even_less_exp import EvenLessExpRunner
import time
import process_database
import numpy as np
from experiment import ExperimentRunner
import parse_data
from CorrespondMetrics import CorrespondMetrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Dense Attacks!')
    parser.add_argument('-points', type=str, default="grid",
                        help='cali_50, grid, dg, crg, nh, boat')

    parser.add_argument('-p', type=float, default=1,
                        help='percentage of queries')
    parser.add_argument('-dist', type=str, default="uniform",
                        help='beta, gaussian, uniform')
    parser.add_argument('-N0', type=int, default=30,
                        help='N0')
    parser.add_argument('-N1', type=int, default=30,
                        help='N1')

    args = parser.parse_args()

    N0 = args.N0
    N1 = args.N1
    N2 = N0
    points, map_to_original = parse_data.extract_data(args.points, args.N0, args.N1)

    # Extract (i, j) coordinates
    coordinates = list(map_to_original.values())

    parse_data.plot_original_data(coordinates)

    '''Fetch response data'''
    print("Getting Responses")
    # get all the queries that can generate all the response
    if "nh" in args.points or "crg" in args.points:
        responses = process_database.get_responses_no_vals_3D(points, map_to_original, N0, N1, N2)

    else:
        responses = process_database.get_responses_no_vals(points, map_to_original, N0, N1)

    print("Sampling Responses")
    unique_rs = set()

    print(int(len(responses) * args.p / 100.0))
    # get the sampled query
    if args.dist == "uniform":
        new_responses = process_database.sample_uniform(responses, int(len(responses) * args.p / 100.0))
    elif args.dist == "beta":
        new_responses = process_database.sample_beta(responses, int(len(responses) * args.p / 100.0))
    elif args.dist == "gaussian":
        new_responses = process_database.sample_gaussian(responses, int(len(responses) * args.p / 100.0))
    else:
        print("I don't know that distribution")
        exit()

    if not ("nh" in args.points or "crg" in args.points or "3d" in args.points):
        new_responses, unique_rs = process_database.get_actual_query_resps_after_sampling(new_responses, points,
                                                                                          map_to_original)
    else:
        new_responses, unique_rs = process_database.get_actual_resps_after_sampling_3D(new_responses, points,
                                                                                       map_to_original)

    runner = ExperimentRunner(points, map_to_original, new_responses)
    even_less_runner = EvenLessExpRunner(new_responses, args.points)

    # Configure experiment parameters
    distance_config = [
        ('reciprocal', {}),
    ]

    reduction_config = [
        ('tsne', {'perplexity': 50, 'method': 'exact'}),
    ]

    '''Run experiments'''
    time1 = time.time()
    results = runner.classic_run(new_responses, distance_config, reduction_config)
    time2 = time.time()
    print("Time taken for mine: ", time2 - time1)
    metrics = CorrespondMetrics(map_to_original, runner.point_to_index, N0=N0, N1=N1)

    evaluation = {}
    for method_name, coords in results.items():
        evaluation[method_name] = metrics.evaluate(coords)

    # Print evaluation results
    for method_name, metrics in evaluation.items():
        print(f"Method: {method_name}")
        for metric_name, value in metrics.items():
            print(f"- {metric_name}: {value:.4f}")

    # '''
    # even less
    # '''
    time3 = time.time()
    even_less_results = even_less_runner.run_ori()
    time4 = time.time()
    print("Time taken for even less: ", time4 - time3)

    # Process results for the "even less" method
    if even_less_results:
        # Create an array with the same size as original coordinates
        even_less_coords = np.zeros((len(runner.point_to_index), 2))

        # Populate coordinates using the point_to_index map
        for points, coords in even_less_results.items():
            for point in points:
                index = runner.point_to_index[point]
                even_less_coords[index] = coords
        # print("Even Less Results: ", even_less_coords)

        metrics = CorrespondMetrics(map_to_original, runner.point_to_index, N0=N0, N1=N1)

        # Evaluate results for the "even less" method
        even_less_metrics = metrics.evaluate(even_less_coords)
        # even_less_runner.plot_result(even_less_coords)
        print("\nEven Less Method Metrics:")
        for metric_name, value in even_less_metrics.items():
            print(f"- {metric_name}: {value:.4f}")


