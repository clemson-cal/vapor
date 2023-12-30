#include <string>
#include <vector>
#include "app_parse.hpp"
#include "app_print.hpp"
#include "core_array.hpp"
#include "hdf5_array.hpp"
#include "hdf5_native.hpp"
#include "hdf5_repr.hpp"
#include "hdf5_vector.hpp"
#include "visit_struct.hpp"




/**
 * Read the full contents of a file to a string.
 *
 * An empty string is returned if the load fails for any reason.
 * 
 */
static inline std::string readfile(const char *filename)
{
    FILE* infile = fopen(filename, "r");
    auto str = std::string();

    while (infile) {
        auto c = fgetc(infile);
        if (c == EOF) {
            break;
        } else {
            str.push_back(c);
        }
    }
    fclose(infile);
    return str;
}




/**
 * Represents a task to be performed at a regular interval
 * 
 */
struct task_state_t
{
    int number = 0;
    double last_time = -1.0;
    double interval = 0.0;

    bool should_be_performed(double t, bool force=false)
    {
        if (last_time == -1.0) {
            last_time = t;
            return true;
        }
        else if (interval > 0.0 && t > last_time + interval) {
            number += 1;
            last_time += interval;
            return true;
        }
        else if (force) {
            number += 1;
            return true;
        }
        else {
            return false;
        }
    }
};

VISITABLE_STRUCT(task_state_t, number, last_time, interval);




/**
 * A data structure holding all of the task states managed by a driver
 * 
 */
struct task_states_t
{
    task_state_t checkpoint;
    task_state_t timeseries;
    task_state_t diagnostic;
};
VISITABLE_STRUCT(task_states_t, checkpoint, timeseries, diagnostic);




template<class Simulation>
int run(int argc, const char **argv, Simulation sim)
{
    auto timeseries_data = std::vector<double>();
    auto tasks = task_states_t();
    auto checkpoint = [&] (const auto& state)
    {
        if (tasks.checkpoint.should_be_performed(state.time))
        {
            auto fname = vapor::format("chkpt.%04d.h5", tasks.checkpoint.number);
            auto h5f = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            visit_struct::for_each(state, [h5f] (auto name, const auto& val)
            {
                vapor::hdf5_write(h5f, name, val);
            });
            vapor::hdf5_write(h5f, "config", sim.config);
            vapor::hdf5_write(h5f, "timeseries", timeseries_data);
            vapor::hdf5_write(h5f, "tasks", tasks);
            vapor::print(vapor::format("write checkpoint %s\n", fname.data));
            H5Fclose(h5f);
        }
    };


    auto timeseries = [&] (const auto& state)
    {
        if (tasks.timeseries.should_be_performed(state.time))
        {
            timeseries_data.push_back(state.time);
        }
    };

    auto diagnostic = [&] (const auto& state)
    {
    };

    tasks.checkpoint.interval = 0.5; // for example

    vapor::set_from_key_vals(sim.config, readfile("session.cfg").data());
    vapor::set_from_key_vals(sim.config, argc, argv);
    vapor::print(sim.config);
    vapor::print("\n");

    {
        FILE *session_cfg = fopen("session.cfg", "w");
        vapor::print(sim.config, session_cfg);
        fclose(session_cfg);
    }

    auto state = sim.initial_state();

    while (sim.should_continue(state))
    {
        timeseries(state);
        diagnostic(state);
        checkpoint(state);
        sim.update(state);
        vapor::print(sim.status_message(state));
        vapor::print("\n");
    }
    timeseries(state);
    diagnostic(state);
    checkpoint(state);

    return 0;
}




struct Simulation
{
    struct Config
    {
        double tfinal = 1.0;
        int num_zones = 100;
    };

    struct State
    {
        double time;
        int iteration;
        vapor::shared_array_t<1, double> u;
    };

    State initial_state() const
    {
        return State{
            0.0, 0, vapor::zeros<double>(vapor::uvec(100)).cache()
        };
    }
    void update(State& state) const
    {
        state.time += 0.0085216;
        state.iteration += 1;
    }
    bool should_continue(const State& state) const
    {
        return state.time < config.tfinal;
    }
    vapor::vec_t<char, 256> status_message(const State& state) const
    {
        return vapor::format("[%04d] t=%lf", state.iteration, state.time);
    }
    // auto post_process(const State& state) const
    // {
    //     auto p = state.u.map(cons_to_prim).cache(executor);
    // }
 
    vapor::cpu_executor_t executor;
    Config config;
};
VISITABLE_STRUCT(Simulation::Config, tfinal, num_zones);
VISITABLE_STRUCT(Simulation::State, time, iteration, u);




int main(int argc, const char **argv)
{
    try {
        return run(argc, argv, Simulation());
    }
    catch (const std::exception& e) {
        printf("[error]: %s\n", e.what());
    }
    return 0;
}
