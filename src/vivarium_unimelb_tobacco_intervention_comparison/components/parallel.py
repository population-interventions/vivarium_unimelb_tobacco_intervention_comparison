"""Run multiple simulations in parallel."""

import datetime
import itertools
import logging
import multiprocessing
import queue
import pickle
import signal
import traceback

import vivarium.framework.configuration as config
import vivarium.framework.engine as engine
import vivarium.framework.plugins as plugins


def fails_to_pickle(item):
    """
    Check whether an object can be serialised ("pickled"), as is required for
    simulation arguments when running simulations in parallel.

    See the ``pickle`` documentation for
    `Python 3 <https://docs.python.org/3/library/pickle.html>`__ for a list of
    types that can be pickled.
    """
    logger = logging.getLogger(__name__)

    def try_iter(value):
        if isinstance(value, dict):
            return value
        else:
            try:
                return range(len(value))
            except TypeError:
                return None

    invalid_paths = []

    def descend_into(value, path):
        try:
            pickle.dumps(value)
        except (pickle.PicklingError, TypeError) as e:
            seq = try_iter(value)
            if seq is not None:
                for i in seq:
                    descend_into(value[i], path + [i])
            else:
                invalid_paths.append((path, value))

    descend_into(item, [])
    if invalid_paths:
        for (path, value) in invalid_paths:
            if path[0] == 2:
                path_str = "][".join(repr(p) for p in path[1:])
                msg = "Invalid value: extra[{}] = {}".format(path_str, value)
            else:
                msg = "Invalid value: {}".format(value)
            logger.error(msg)
        return True

    return False


def run_in_parallel(func, iterable, n_proc):
    """
    Perform multiple simulations in parallel by spawning multiple processes.

    :param func: The function that performs a single simulation.
    :param iterable: A sequence of simulation arguments, represented as tuples
        and *unpacked* before passing to ``func`` (i.e., ``func(*args)``).
    :param n_proc: The number of processes to spawn.

    :returns: ``True`` if all jobs were successfully completed (i.e., each
        process terminated with an exit code of ``0``).
    """
    # Note: we avoid using multiprocessing.Pool because it does not handle
    # KeyboardInterrupt exceptions correctly. For details, see:
    # http://bryceboe.com/2012/02/14/python-multiprocessing-pool-and-keyboardinterrupt-revisited/
    logger = logging.getLogger(__name__)
    job_q = multiprocessing.Queue()
    workers = []

    def apply_func(job_q):
        # Ignore the signal that raises KeyboardInterrupt exceptions; the main
        # loop will handle this exception and ensure each process terminates.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        while not job_q.empty():
            try:
                args = job_q.get(block=False)
                func(*args)
            except queue.Empty:
                pass
            except Exception:
                print(traceback.format_exc())

    n_job = 0
    for args in iterable:
        if fails_to_pickle(args):
            logger.error("Encountered invalid arguments, terminating")
            return
        n_job += 1
        try:
            job_q.put(args, block=False)
        except queue.Full:
            logger.error("Could not add job {} to the queue".format(n_job))
            return

    try:
        # Start the worker processes.
        if n_proc > n_job:
            # Spawn no more processes than there are jobs
            n_proc = n_job
        logger.info("Spawning {} workers for {} jobs".format(n_proc, n_job))
        for i in range(n_proc):
            proc = multiprocessing.Process(target=apply_func, args=[job_q])
            workers.append(proc)
            proc.start()
        # Wait for each worker to finish. Without this loop, we jump straight
        # to the finally clause and the KeyboardInterrupt handler (below) is
        # never triggered.
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        # Force each worker to terminate.
        logger.info("Received CTRL-C, terminating {} workers".format(n_proc))
        for worker in workers:
            worker.terminate()
    except Exception as e:
        print(e)
    finally:
        all_good = True
        # Wait for each worker to finish.
        for ix, worker in enumerate(workers):
            worker.join()
            # Note: worker.exitcode should be 0 if it completed successfully.
            # It will be -N if it was terminated by signal N.
            # It will apparently be 1 if an exception was raised.
            all_good = all_good and worker.exitcode == 0
            if worker.exitcode != 0:
                msg = "Worker {} exit code: {}"
                logger.info(msg.format(ix, worker.exitcode))
        return all_good




def initialise_simulation_from_specification_config(model_specification):
    """
    Construct a simulation object from a model specification.

    :param model_specification: The simulation specification (``ConfigTree``).
    """
    plugin_config = model_specification.plugins
    component_config = model_specification.components
    simulation_config = model_specification.configuration

    plugin_manager = plugins.PluginManager(plugin_config)
    component_config_parser = plugin_manager.get_plugin('component_configuration_parser')
    components = component_config_parser.get_components(component_config)

    simulation = engine.SimulationContext(simulation_config, components,
                                          plugin_manager)

    return simulation


def run_nth_draw(model_specification_file, draw_number):
    """
    Run a model simulation for a specific draw number.

    :param model_specification_file: The YAML model specification file.
    :param draw_number: The draw number to select for rates and values that
        have multiple draws.
    """
    logger = logging.getLogger(__name__)
    logger.info('{} Simulating draw #{} for {} ...'.format(
        datetime.datetime.now().strftime("%H:%M:%S"),
        draw_number, model_specification_file))
    spec = config.build_model_specification(model_specification_file)
    spec.configuration.input_data.input_draw_number = draw_number

    simulation = initialise_simulation_from_specification_config(spec)
    simulation.setup()

    metrics, final_state = engine.run(simulation)
    logger.info('{} Simulation for draw #{} complete'.format(
        datetime.datetime.now().strftime("%H:%M:%S"),
        draw_number))

    return metrics, final_state


def run_many(spec_files, num_draws, num_procs):
    """
    Run a number of model simulations in serial or in parallel.

    :param spec_files: A list of model specification files.
    :param num_draws: The number of draws for which simulations will be run,
        including draw number zero (i.e., the expected values).
    :param num_procs: The number of processes to spawn in order to run these
        simulations; set this to values greater than 1 to run multiple
        simulations in parallel.
    :returns: ``True`` if the simulations completed successfully, otherwise
        ``False``.
    """
    if num_procs < 1:
        raise ValueError('Invalid number of processes: {}'.format(num_procs))
    elif num_procs == 1:
        # Run the simulations serially.
        for spec_file in spec_files:
            for draw in range(num_draws + 1):
                metrics, final_state = run_nth_draw(spec_file, draw)
        return True
    else:
        # Run the simulations in parallel.
        args_iter = itertools.product(spec_files, range(num_draws + 1))
        return run_in_parallel(run_nth_draw, args_iter, num_procs)
