#include "app_print.hpp"
#include "app_parse.hpp"
#include "core_array.hpp"
#include "visit_struct.hpp"
#include "hdf5_repr.hpp"
#include "hdf5_native.hpp"
#include "hdf5_array.hpp"




template<typename Config, typename State>
class SimulationBase
{
public:

    virtual void update(State& state) const = 0;
    virtual bool should_continue(const State& state) const = 0;
    virtual vapor::vec_t<char, 256> status_message(const State& state) const = 0;

    Config& configuration()
    {
        return config;
    }

protected:
    Config config;
};




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




class Simulation : public SimulationBase<Config, State>
{
public:
    auto initial_state() const
    {
        return State{
            0.0, 0, vapor::zeros<double>(vapor::uvec(100)).cache()
        };
    }
    void update(State& state) const override
    {
        state.time += 0.1;
        state.iteration += 1;
    }
    bool should_continue(const State& state) const override
    {
        return state.time < config.tfinal;
    }
    vapor::vec_t<char, 256> status_message(const State& state) const override
    {
        return vapor::message("[%04d] t=%lf", state.iteration, state.time);
    }
private:
    vapor::cpu_executor_t executor;
};




template<class Simulation>
int run(int argc, const char **argv, Simulation sim)
{
    vapor::set_from_key_vals(sim.configuration(), argc, argv);
    vapor::print(sim.configuration());
    vapor::print("\n");

    auto state = sim.initial_state();

    while (sim.should_continue(state))
    {
        sim.update(state);
        printf("%s\n", sim.status_message(state).data);
    }
    vapor::hdf5_write_file("chkpt.0000.h5", state);
    return 0;
}




int main(int argc, const char **argv)
{
    try {
        return run(argc, argv, Simulation());
    }
    catch (const std::exception& e) {
        printf("[error]: %s\n", e.what());
    }
}
