import matplotlib.pyplot as plt
import argparse
from even_less_exp import EvenLessExpRunner
import time
import process_database
import numpy as np
from experiment import ExperimentRunner
import parse_data
import range_attack
from metrics_3d import CorrespondMetrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run High Dimension Attacks!')
    parser.add_argument('-points', type=str, default="grid_nd",
                        help='grid_nd, obesity_3d, credit_3d')

    parser.add_argument('-p', type=float, default=1,
                        help='percentage of queries')
    parser.add_argument('-dist', type=str, default="uniform",
                        help='beta, gaussian, uniform')
    parser.add_argument('-N', type=int, default=16,
                        help='N')
    parser.add_argument('-dim', type=int, default=3,
                        help='dim')

    args = parser.parse_args()

    N0 = args.N
    N1 = args.N
    N2 = args.N
    dim = args.dim
    points, map_to_original = parse_data.extract_data(args.points, N0, args.dim)

    coordinates = list(map_to_original.values())


    '''Fetch response data'''
    print("Getting Responses")
    # get all the queries that can generate all the response
    if "nh" in args.points or "crg" in args.points or "3d" in args.points:
        responses = process_database.get_responses_no_vals_3D(points, map_to_original, N0, N1, N2)
    elif args.points == "grid_nd":
        responses = process_database.get_responses_no_vals_nd(points, map_to_original, N0, dim)
    else:
        responses = process_database.get_responses_no_vals(points, map_to_original, N0, N1)

    print("Sampling Responses")
    unique_rs = set()

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

    if not ("nh" in args.points or "crg" in args.points or "3d" in args.points or "grid_nd" in args.points):
        new_responses, unique_rs = process_database.get_actual_query_resps_after_sampling(new_responses, points,
                                                                                          map_to_original)
    elif args.points == "grid_nd":
        new_responses, unique_rs = process_database.get_actual_query_resps_after_sampling_nd(new_responses, points,
                                                                                             map_to_original)
    else:
        new_responses, unique_rs = process_database.get_actual_query_resps_after_sampling_3D(new_responses, points,
                                                                                             map_to_original)

    runner = ExperimentRunner(points, map_to_original, new_responses)
    even_less_runner = EvenLessExpRunner(new_responses, args.points)

    # Configure experiment parameters
    distance_config = [
        ('reciprocal', {}),
    ]

    reduction_config = [
        ('tsne', {'perplexity': 50, 'n_components': args.dim, 'method': 'exact'})
    ]

    '''Run experiments'''
    time1 = time.time()
    results = runner.classic_run(new_responses, distance_config, reduction_config)
    time2 = time.time()
    print("Time taken for mine: ", time2 - time1)

    tuple_N = (N0,) * dim
    metrics = CorrespondMetrics(map_to_original, runner.point_to_index, tuple_N)

    evaluation = {}
    for method_name, coords in results.items():
        evaluation[method_name] = metrics.evaluate(coords)

    for method_name, metrics in evaluation.items():
        print(f"Method: {method_name}")
        for metric_name, value in metrics.items():
            print(f"- {metric_name}: {value:.4f}")


    # '''
    # even less
    # '''
    time3 = time.time()
    even_less_results = even_less_runner.run_ori(dim=dim)
    time4 = time.time()
    print("Time taken for even less: ", time4 - time3)

    # Process results for the "even less" method
    if even_less_results:
        even_less_coords = np.zeros((len(runner.point_to_index), dim))

        for points, coords in even_less_results.items():
            for point in points:
                index = runner.point_to_index[point]
                even_less_coords[index] = coords


        metrics = CorrespondMetrics(map_to_original, runner.point_to_index, tuple_N)

        # Evaluate results for the "even less" method
        even_less_metrics = metrics.evaluate(even_less_coords)
        print("\nEven Less Method Metrics:")
        for metric_name, value in even_less_metrics.items():
            print(f"- {metric_name}: {value:.4f}")