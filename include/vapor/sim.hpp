/**
================================================================================
Copyright 2023 - 2024, Jonathan Zrake

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
================================================================================
*/




#pragma once
#define VAPOR_STD_MAP
#define VAPOR_STD_STRING
#define VAPOR_STD_VECTOR
#include <chrono>
#include <map>
#include <string>
#include <vector>
#include "hdf5/hdf5_array.hpp"
#include "hdf5/hdf5_map.hpp"
#include "hdf5/hdf5_native.hpp"
#include "hdf5/hdf5_repr.hpp"
#include "hdf5/hdf5_vector.hpp"
#include "vapor/array.hpp"
#include "vapor/executor.hpp"
#include "vapor/parse.hpp"
#include "vapor/print.hpp"
#include "visit_struct/visit_struct.hpp"

namespace vapor {
    template<class F>
    double time_call(int num_calls, F f);
    static inline std::string readfile(const char *filename);
    struct task_state_t;
    struct task_states_t;
    template<class C, class S, class D> class Simulation;
    template<class Config, class State, class DiagnosticData>
    int run(int argc, const char **argv, Simulation<Config, State, DiagnosticData>& sim);
}




/**
 * Simulation base class
 *
 */
template<class C, class S, class D>
class vapor::Simulation
{
public:
    using Config = C;
    using State = S;
    using DiagnosticData = D;

    /**
     * Return a time for use by the simulation driver
     *
     * The returned value is used by the driver to check whether it is time to
     * perform simulation tasks.
     */
    virtual double get_time(const State& state) const = 0;

    /**
     * Return an iteration number for use by the driver
     *
     * The iteration number should be incremented once each time the update
     * function is called
     */
    virtual vapor::uint get_iteration(const State& state) const = 0;

    /**
     * Generate an initial state for the simulation
     *
     * Must be overriden by derived classes
     */
    virtual void initial_state(State& state) const = 0;

    /**
     * Update the simulation state by one iteration
     *
     * Must be overridden by derived classes
     */
    virtual void update(State& state) const = 0;

    /**
     * Test whether the simulation has reached a desired stopping point
     *
     * Must be overridden by derived classes
     */
    virtual bool should_continue(const State& state) const = 0;

    /**
     * Return the number of updates between status messages
     *
     * The performance measurement is averaged over a batch, so tends to be
     * more accurate when there are more updates per batch.
     *
     * May be overridden by derived classes 
     */
    virtual vapor::uint updates_per_batch() { return 10; }

    /**
     * Return the time between checkpoint task recurrences
     *
     * May be overridden by derived classes
     */
    virtual double checkpoint_interval() const { return 0.0; }

    /**
     * Return the time between timeseries task recurrences
     *
     * Will likely be overridden by derived classes
     */
    virtual double timeseries_interval() const { return 0.0; }

    /**
     * Return the time between diagnostic task recurrences
     *
     * Will likely be overridden by derived classes
     */
    virtual double diagnostic_interval() const { return 0.0; }

    /**
     * Return a status message to be printed by the driver
     *
     * May be overridden by derived classes
     */
    virtual vapor::vec_t<char, 256> status_message(const State& state, double secs_per_update) const
    {
        return vapor::format("[%04d] t=%lf %lf sec/update",
            get_iteration(state),
            get_time(state), secs_per_update);
    }

    /**
     * Return integers identifying the timeseries measurements to be made
     *
     * The integers returned must be valid keys to the get_timeseries_name and
     * compute_timeseries_sample functions below. They could be hard-coded, or
     * be collected from the user configuration.
     * 
     */
    virtual std::vector<vapor::uint> get_timeseries_cols() const
    {
        return {};
    }

    /**
     * Return a short name for one of the provided timeseries measurements
     *
     */
    virtual const char* get_timeseries_name(vapor::uint column) const
    {
        return nullptr;
    }

    /**
     * Compute and return a number from the simulation state
     *
     */
    virtual double compute_timeseries_sample(const State& state, vapor::uint column) const
    {
        return {};
    }

    /**
     * Return integers identifying the diagnostic fields to be computed
     *
     * The integers returned must be valid keys to the get_diagnostic_name and
     * compute_diagnostic functions below. They could be hard-coded, or be
     * collected from the user configuration.
     * 
     */
    virtual std::vector<vapor::uint> get_diagnostic_cols() const
    {
        return {};
    }

    /**
     * Return a short name for one of the provided diagnostic measurements
     *
     */
    virtual const char* get_diagnostic_name(vapor::uint column) const
    {
        return nullptr;
    }

    /**
     * Compute and return a diagnsotic field from the simulation state
     *
     */
    virtual DiagnosticData compute_diagnostic(const State& state, vapor::uint column) const
    {
        return {};
    }

    /**
     * Returns a non-const reference to the configuration instance
     *
     * Should not be overriden by derived classes
     * 
     */
    Config& get_config() { return config; }

    /**
     * Returns a const reference to the configuration instance
     *
     * Should not be overriden by derived classes
     * 
     */
    const Config& get_config() const { return config; }

protected:
    Config config;
};




/**
 * Execute and time a function call
 *
 * The function is called the given number of times. Returns the number of
 * seconds per call as a double.
 */
template<class F>
double vapor::time_call(int num_calls, F f)
{
    using namespace std::chrono;
    auto t1 = high_resolution_clock::now();
    for (int n = 0; n < num_calls; ++n) {
        f();
    }
    auto t2 = high_resolution_clock::now();
    auto delta = duration_cast<duration<double>>(t2 - t1);
    return delta.count() / num_calls;
}




/**
 * Read the full contents of a file to a string.
 *
 * An empty string is returned if the load fails for any reason.
 * 
 */
std::string vapor::readfile(const char *filename)
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
struct vapor::task_state_t
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
VISITABLE_STRUCT(vapor::task_state_t, number, last_time, interval);




/**
 * A data structure holding all of the task states managed by a driver
 * 
 */
struct vapor::task_states_t
{
    task_state_t checkpoint;
    task_state_t timeseries;
    task_state_t diagnostic;
};
VISITABLE_STRUCT(vapor::task_states_t, checkpoint, timeseries, diagnostic);




template<class Config, class State, class DiagnosticData>
int vapor::run(int argc, const char **argv, Simulation<Config, State, DiagnosticData>& sim)
{
    auto state = State();
    auto tasks = task_states_t();
    auto timeseries_data = std::map<std::string, std::vector<double>>();
    auto checkpoint = [&] (const auto& state)
    {
        if (tasks.checkpoint.should_be_performed(sim.get_time(state)))
        {
            auto fname = vapor::format("chkpt.%04d.h5", tasks.checkpoint.number);
            auto h5f = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            vapor::hdf5_write(h5f, "state", state);
            vapor::hdf5_write(h5f, "config", sim.get_config());
            vapor::hdf5_write(h5f, "timeseries", timeseries_data);
            vapor::hdf5_write(h5f, "tasks", tasks);
            H5Fclose(h5f);
            printf("write checkpoint %s\n", fname.data);
        }
    };

    auto timeseries = [&] (const auto& state)
    {
        if (tasks.timeseries.should_be_performed(sim.get_time(state)))
        {
            for (auto col : sim.get_timeseries_cols())
            {
                auto name = sim.get_timeseries_name(col);
                auto sample = sim.compute_timeseries_sample(state, col);
                timeseries_data[name].push_back(sample);
            }
            printf("record timeseries entry\n");
        }
    };

    auto diagnostic = [&] (const auto& state)
    {
        if (tasks.diagnostic.should_be_performed(sim.get_time(state)))
        {
            auto fname = vapor::format("diagnostic.%04d.h5", tasks.diagnostic.number);
            auto h5f = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            for (auto col : sim.get_diagnostic_cols())
            {
                auto name = sim.get_diagnostic_name(col);
                auto field = sim.compute_diagnostic(state, col);
                vapor::hdf5_write(h5f, name, field);
            }
            H5Fclose(h5f);
            printf("write diagnostic file %s\n", fname.data);
        }
    };

    if (argc > 1 && strstr(argv[1], ".h5"))
    {
        auto h5f = H5Fopen(argv[1], H5P_DEFAULT, H5P_DEFAULT);
        vapor::hdf5_read(h5f, "state", state);
        vapor::hdf5_read(h5f, "config", sim.get_config());
        vapor::hdf5_read(h5f, "timeseries", timeseries_data);
        vapor::hdf5_read(h5f, "tasks", tasks);
        vapor::set_from_key_vals(sim.get_config(), argc - 1, argv + 1);
        H5Fclose(h5f);
        printf("read checkpoint %s\n", argv[1]);
    }
    else
    {
        vapor::set_from_key_vals(sim.get_config(), readfile("session.cfg").data());
        vapor::set_from_key_vals(sim.get_config(), argc, argv);
        vapor::print_to_file(sim.get_config(), "session.cfg");
        sim.initial_state(state);
    }
    tasks.checkpoint.interval = sim.checkpoint_interval();
    tasks.timeseries.interval = sim.timeseries_interval();
    tasks.diagnostic.interval = sim.diagnostic_interval();

    vapor::print(sim.get_config());
    vapor::print("\n");

    while (sim.should_continue(state))
    {
        timeseries(state);
        diagnostic(state);
        checkpoint(state);

        auto secs = time_call(sim.updates_per_batch(), [&sim, &state]
        {
            sim.update(state);
        });

        vapor::print(sim.status_message(state, secs));
        vapor::print("\n");
    }
    timeseries(state);
    diagnostic(state);
    checkpoint(state);

    return 0;
}
