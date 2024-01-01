#include <chrono>
#include <map>
#include <string>
#include <vector>
#include "vapor/parse.hpp"
#include "vapor/print.hpp"
#include "vapor/array.hpp"
#include "hdf5/hdf5_array.hpp"
#include "hdf5/hdf5_map.hpp"
#include "hdf5/hdf5_native.hpp"
#include "hdf5/hdf5_repr.hpp"
#include "hdf5/hdf5_vector.hpp"
#include "visit_struct/visit_struct.hpp"




// #include <sys/stat.h> // for directory creation

// static void makedir(const char *dir)
// {
//     char tmp[256];
//     snprintf(tmp, sizeof(tmp), "%s", dir);
//     size_t len = strlen(tmp);

//     if (tmp[len - 1] == '/') {
//         tmp[len - 1] = 0;
//     }
//     for (char* p = tmp + 1; *p; p++) {
//         if (*p == '/') {
//             *p = 0;
//             mkdir(tmp, S_IRWXU);
//             *p = '/';
//         }
//     }
//     mkdir(tmp, S_IRWXU);
// }




/**
 * Execute and time a function call
 *
 * The function is called the given number of times. Returns the number of
 * seconds per call as a double.
 */
template<class F>
double time_call(int num_calls, F f)
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




/**
 * Simulation base class
 *
 * Provides or requires 
 */
template<class C, class S>
class Simulation
{
public:
    using Config = C;
    using State = S;

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
    virtual State initial_state() const = 0;

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
     * Returns a non-const reference to the configuration instance
     *
     * Should not be overriden by derived classes
     */
    Config& get_config() { return config; }

    /**
     * Returns a const reference to the configuration instance
     *
     * Should not be overriden by derived classes
     */
    const Config& get_config() const { return config; }

protected:
    Config config;
};




template<class Config, class State>
int run(int argc, const char **argv, Simulation<Config, State>& sim)
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
            // for example
            timeseries_data["time"].push_back(sim.get_time(state));
            printf("record timeseries entry\n");
        }
    };

    auto diagnostic = [&] (const auto& state)
    {
        if (tasks.diagnostic.should_be_performed(sim.get_time(state)))
        {
            printf("write diagnostic file\n");
        }
    };

    timeseries_data["time"] = {}; // for example

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
        state = sim.initial_state();
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




#include <unistd.h> // usleep, for demo simulation class

struct Config
{
    int num_zones = 100;
    double tfinal = 1.0;
    double cpi = 0.0;
};
VISITABLE_STRUCT(Config, num_zones, tfinal, cpi);

struct State
{
    double time;
    int iteration;
    vapor::memory_backed_array_t<1, double, std::shared_ptr> u;
};
VISITABLE_STRUCT(State, time, iteration, u);

class DemoSimulation : public Simulation<Config, State>
{
public:
    double get_time(const State& state) const override
    {
        return state.time;
    }
    virtual vapor::uint get_iteration(const State& state) const override
    {
        return state.iteration;
    }
    State initial_state() const override
    {
        return State{
            0.0,
            0,
            vapor::zeros<double>(vapor::uvec(config.num_zones)).cache(executor, allocator)
        };
    }
    void update(State& state) const override
    {
        state.time += 0.0085216;
        state.iteration += 1;
        usleep(10000);
    }
    bool should_continue(const State& state) const override
    {
        return state.time < config.tfinal;
    }
    double checkpoint_interval() const override
    { 
        return config.cpi;
    }
private:
    vapor::cpu_executor_t executor;
    vapor::shared_ptr_allocator_t allocator;
};




int main(int argc, const char **argv)
{
    try {
        auto sim = DemoSimulation();
        return run(argc, argv, sim);
    }
    catch (const std::exception& e) {
        printf("[error]: %s\n", e.what());
    }
    return 0;
}
