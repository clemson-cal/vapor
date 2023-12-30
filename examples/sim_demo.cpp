#include <chrono>
#include <string>
#include <vector>
#include <unistd.h> // usleep, for demo simulation class
// #include <sys/stat.h> // for directory creation
#include "app_parse.hpp"
#include "app_print.hpp"
#include "core_array.hpp"
#include "hdf5_array.hpp"
#include "hdf5_native.hpp"
#include "hdf5_repr.hpp"
#include "hdf5_vector.hpp"
#include "visit_struct.hpp"




// TODO
// 
// [ ] multi-threaded execution
// [ ] use of executor base class (see if affects performance)
// [ ] run directory feature
// [ ] diagnostic and time series outputs
// [ ] crash / safety mode feature
// [x] simulation base class




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
 * Execute and time function call
 *
 * The function is called the given number of times. Returns the number of
 * seconds per call as a double.
 */
template<class F>
double time_call(F f, int num_calls)
{
    auto t1 = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < num_calls; ++n)
    {
        f();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
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
     * Return a status message to be printed by the driver
     *
     * Will likely be overridden by derived classes
     */
    virtual vapor::vec_t<char, 256> status_message(const State& state, double secs_per_call) const
    {
        return vapor::format("[%04d] t=%lf %lf sec/iter", get_iteration(state), get_time(state), secs_per_call);
    }

    /**
     * Returns a non-const reference to the configuration instance
     *
     * Should not be overriden by derived classes
     */
    virtual Config& get_config() { return config; }

    /**
     * Returns a const reference to the configuration instance
     *
     * Should not be overriden by derived classes
     */
    virtual const Config& get_config() const { return config; }

protected:
    Config config;
};




template<class Config, class State>
int run(int argc, const char **argv, Simulation<Config, State>& sim)
{
    auto timeseries_data = std::vector<double>();
    auto tasks = task_states_t();
    auto checkpoint = [&] (const auto& state)
    {
        if (tasks.checkpoint.should_be_performed(sim.get_time(state)))
        {
            auto fname = vapor::format("chkpt.%04d.h5", tasks.checkpoint.number);
            auto h5f = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            visit_struct::for_each(state, [h5f] (auto name, const auto& val)
            {
                vapor::hdf5_write(h5f, name, val);
            });
            vapor::hdf5_write(h5f, "config", sim.get_config());
            vapor::hdf5_write(h5f, "timeseries", timeseries_data);
            vapor::hdf5_write(h5f, "tasks", tasks);
            printf("write checkpoint %s\n", fname.data);
            H5Fclose(h5f);
        }
    };

    auto timeseries = [&] (const auto& state)
    {
        if (tasks.timeseries.should_be_performed(sim.get_time(state)))
        {
            timeseries_data.push_back(sim.get_time(state));
        }
    };

    auto diagnostic = [&] (const auto& state)
    {
        // todo
    };

    tasks.checkpoint.interval = 0.5; // for example
    auto state = State();

    if (argc > 1 && strstr(argv[1], ".h5"))
    {
        printf("read checkpoint %s\n", argv[1]);
        auto h5f = H5Fopen(argv[1], H5P_DEFAULT, H5P_DEFAULT);
        visit_struct::for_each(state, [h5f] (auto name, auto& val)
        {
            vapor::hdf5_read(h5f, name, val);
        });
        vapor::hdf5_read(h5f, "config", sim.get_config());
        vapor::hdf5_read(h5f, "timeseries", timeseries_data);
        vapor::hdf5_read(h5f, "tasks", tasks);
        vapor::set_from_key_vals(sim.get_config(), argc - 1, argv + 1);
        H5Fclose(h5f);
    }
    else
    {
        vapor::set_from_key_vals(sim.get_config(), readfile("session.cfg").data());
        vapor::set_from_key_vals(sim.get_config(), argc, argv);
        vapor::print_to_file(sim.get_config(), "session.cfg");
        state = sim.initial_state();
    }
    vapor::print(sim.get_config());
    vapor::print("\n");

    while (sim.should_continue(state))
    {
        timeseries(state);
        diagnostic(state);
        checkpoint(state);

        auto secs = time_call([&sim, &state] { sim.update(state); }, 10);

        vapor::print(sim.status_message(state, secs));
        vapor::print("\n");
    }
    timeseries(state);
    diagnostic(state);
    checkpoint(state);

    return 0;
}




struct Config
{
    double tfinal = 1.0;
    int num_zones = 100;
};
VISITABLE_STRUCT(Config, tfinal, num_zones);

struct State
{
    double time;
    int iteration;
    vapor::shared_array_t<1, double> u;
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
            0.0, 0, vapor::zeros<double>(vapor::uvec(config.num_zones)).cache()
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
