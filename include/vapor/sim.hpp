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
#include <filesystem>
#include <set>
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
    template<class Config, class State, class Product>
    int run(int argc, const char **argv, Simulation<Config, State, Product>& sim);
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
    using Product = D;

    /**
     * Return a name for the simulation class
     *
     * This should also be the name of the executable, because it will be used
     * in generating a usage message from inside the run function.
     */
    virtual const char* name() const { return nullptr; }

    /**
     * Return an author name or names for the simulation
     *
     */
    virtual const char* author() const { return nullptr; }

    /**
     * Return a short description of the simulation
     *
     */
    virtual const char* description() const { return nullptr; }

    /**
     * A filesystem location where simulations outputs should be written
     *
     */
    virtual const char* output_directory() const { return nullptr; }

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
    virtual vapor::uint updates_per_batch() const { return 10; }

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
     * Return the time between product task recurrences
     *
     * Will likely be overridden by derived classes
     */
    virtual double product_interval() const { return 0.0; }

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
    virtual std::set<vapor::uint> get_timeseries_cols() const
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
     * Return integers identifying the product fields to be computed
     *
     * The integers returned must be valid keys to the get_product_name and
     * compute_product functions below. They could be hard-coded, or be
     * collected from the user configuration.
     * 
     */
    virtual std::set<vapor::uint> get_product_cols() const
    {
        return {};
    }

    /**
     * Return a short name for one of the provided product measurements
     *
     */
    virtual const char* get_product_name(vapor::uint column) const
    {
        return nullptr;
    }

    /**
     * Compute and return a product from the simulation state
     *
     */
    virtual Product compute_product(const State& state, vapor::uint column) const
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

    /**
     * Run a simulation to completion, using arguments to main
     *
     * Should not be overridden by derived classes
     */
    int run(int argc, const char **argv)
    {
        return vapor::run(argc, argv, *this);
    }
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
    task_state_t product;
};
VISITABLE_STRUCT(vapor::task_states_t, checkpoint, timeseries, product);




template<class Config, class State, class Product>
int vapor::run(int argc, const char **argv, Simulation<Config, State, Product>& sim)
{
    auto state = State();
    auto tasks = task_states_t();
    auto timeseries_data = std::map<std::string, std::vector<double>>();
    auto output_directory = std::filesystem::path(sim.output_directory() ? sim.output_directory() : ".");
    auto checkpoint = [&] (const auto& state)
    {
        if (tasks.checkpoint.should_be_performed(sim.get_time(state)))
        {
            auto fname = output_directory / vapor::format("chkpt.%04d.h5", tasks.checkpoint.number).data;
            auto h5f = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            vapor::hdf5_write(h5f, "state", state);
            vapor::hdf5_write(h5f, "config", sim.get_config());
            vapor::hdf5_write(h5f, "timeseries", timeseries_data);
            vapor::hdf5_write(h5f, "tasks", tasks);
            H5Fclose(h5f);
            printf("write %s\n", fname.c_str());
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

    auto product = [&] (const auto& state)
    {
        if (tasks.product.should_be_performed(sim.get_time(state)))
        {
            auto fname = output_directory / vapor::format("prods.%04d.h5", tasks.product.number).data;
            auto h5f = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            for (auto col : sim.get_product_cols())
            {
                auto name = sim.get_product_name(col);
                auto field = sim.compute_product(state, col);
                vapor::hdf5_write(h5f, name, field);
            }
            H5Fclose(h5f);
            printf("write %s\n", fname.c_str());
        }
    };

    for (int n = 0; n < argc; ++n)
    {
        if (strcmp(argv[n], "-h") == 0)
        {
            if (auto name = sim.name())
                printf("usage: %s [restart.h5] [key=val...]\n", name);
            if (auto author = sim.author())
                printf("author: %s\n", author);
            if (auto description = sim.description())
                printf("description: %s\n", description);

            {
                uint col = 0;
                while (auto name = sim.get_product_name(col))
                {
                    if (col == 0)
                    {
                        vapor::print("\nsimulation products:\n");
                    }
                    vapor::print(format("%d: %s\n", col, sim.get_product_name(col)));
                    col += 1;
                }
                if (col > 0)
                {
                    vapor::print("\n");
                }
            }

            {
                uint col = 0;
                while (auto name = sim.get_timeseries_name(col))
                {
                    if (col == 0)
                    {
                        vapor::print("\ntime series options:\n");
                    }
                    vapor::print(format("%d: %s\n", col, sim.get_timeseries_name(col)));
                    col += 1;
                }
                if (col > 0)
                {
                    vapor::print("\n");
                }
            }
            return 0;
        }
        else if (argv[n][0] == '-') {
            throw std::runtime_error(vapor::format("unrecognized option %s", argv[n]));
        }
    }
    if (argc > 1 && strstr(argv[1], ".h5"))
    {
        auto h5f = H5Fopen(argv[1], H5P_DEFAULT, H5P_DEFAULT);
        vapor::hdf5_read(h5f, "state", state);
        vapor::hdf5_read(h5f, "config", sim.get_config());
        vapor::hdf5_read(h5f, "timeseries", timeseries_data);
        vapor::hdf5_read(h5f, "tasks", tasks);
        vapor::set_from_key_vals(sim.get_config(), argc - 1, argv + 1);
        H5Fclose(h5f);
        printf("read %s\n", argv[1]);
    }
    else
    {
        vapor::set_from_key_vals(sim.get_config(), readfile("session.cfg").data());
        vapor::set_from_key_vals(sim.get_config(), argc, argv);
        vapor::print_to_file(sim.get_config(), "session.cfg");
        sim.initial_state(state);
    }

    if (auto outdir = sim.output_directory()) {
        printf("write output to %s\n", outdir);
        std::filesystem::create_directories(outdir);
    }
    tasks.checkpoint.interval = sim.checkpoint_interval();
    tasks.timeseries.interval = sim.timeseries_interval();
    tasks.product.interval = sim.product_interval();

    vapor::print("\n");
    vapor::print(sim.get_config());
    vapor::print("\n");

    if (! sim.get_product_cols().empty())
    {
        for (auto col : sim.get_product_cols())
        {
            if (auto name = sim.get_product_name(col))
                print(format("product %d: %s\n", col, name));
            else
                throw std::runtime_error(format("product number %d is not provided", col));
        }
        vapor::print("\n");
    }

    if (! sim.get_timeseries_cols().empty())
    {
        for (auto col : sim.get_timeseries_cols())
        {
            if (auto name = sim.get_timeseries_name(col))
                print(format("timeseries %d: %s\n", col, name));
            else
                throw std::runtime_error(format("timeseries number %d is not provided", col));
        }
        vapor::print("\n");
    }

    while (sim.should_continue(state))
    {
        timeseries(state);
        checkpoint(state);
        product(state);

        auto secs = time_call(sim.updates_per_batch(), [&sim, &state]
        {
            sim.update(state);
        });

        vapor::print(sim.status_message(state, secs));
        vapor::print("\n");
    }
    timeseries(state);
    checkpoint(state);
    product(state);

    return 0;
}
