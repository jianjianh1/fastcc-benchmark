#include "contract.hpp"
#include <taskflow/taskflow.hpp>

class TaskQueue {
  tf::Taskflow taskflow;
  // At the end of the symbolic analysis, this contains the task DAG.
  // The doubles and singles results are held in this class as well.
  //
  // tf::Executor executor{1};
  tf::Executor executor;
  int task_id = 0;
  int num_runs = 0;
  Tensor<densemat> doubles_result = Tensor<densemat>("./T2.tns", true);

public:
  Tensor<densemat> &getDoubles() { return doubles_result; }
  void run() { executor.run(taskflow).wait(); }
  template <class Res, class L, class R>
  void addContraction(Tensor<Res> &result, Tensor<L> &left, Tensor<R> &right,
                      CoOrdinate left_contr, CoOrdinate left_batch,
                      CoOrdinate right_contr, CoOrdinate right_batch) {
    // TODO potentially we can make it faster using the Symbolic result, but
    // that's for later.
    taskflow.emplace([&result, &left, &right, left_contr, left_batch,
                      right_contr, right_batch]() mutable {
      result.fill_values(left, right, left_contr, left_batch, right_contr,
                         right_batch);
      // std::cout << result.get_nonzeros().size() << std::endl;
    });
  }
  template <class Res, class L, class R>
  tf::Task makeTask(Tensor<Res> *result, Tensor<L> &left, Tensor<R> &right,
                    CoOrdinate left_contr, CoOrdinate left_batch,
                    CoOrdinate right_contr, CoOrdinate right_batch,
                    std::string lname, std::string rname) {
    // TODO potentially we can make it faster using the Symbolic result, but
    // that's for later.
    // std::cout << "Queue has " << taskflow.num_tasks() << " tasks" <<
    // std::endl;
    auto task_id = this->task_id++;
    return taskflow
        .emplace([result, &left, &right, left_contr, left_batch, right_contr,
                  right_batch, task_id, lname, rname]() mutable {
          result->fill_values(left, right, left_contr, left_batch, right_contr,
                              right_batch);
        })
        .name(lname + " * " + rname);
  }

  tf::Task updateDoubles(Tensor<densemat> *equation_result,
                         std::string tensor_name = "") {
    auto task_id = this->task_id++;
    return taskflow
        .emplace([equation_result, task_id, this]() mutable {
          this->doubles_result += equation_result;
        })
        .name("Update doubles " + std::to_string(task_id) + "_" + tensor_name);
  }

  double nextGuess() {
    Tensor<densemat> dr2_residual = doubles_result.multiply<densemat>(
        doubles_result, CoOrdinate({}), CoOrdinate({0, 1}), CoOrdinate({}),
        CoOrdinate({0, 1}));
    double doubles_residual = 0.5 * std::sqrt(dr2_residual.reduce());
    //TODO get the denominator evl tensor from the disk and use that element-wise
    Tensor<densemat> dr2_denominator = Tensor<densemat>("./T2.tns", true);
    dr2_residual /= &dr2_denominator;
    doubles_result = dr2_residual;
    return doubles_residual;
  }

  //bool hasConverged() {
  //    auto err = nextGuess();
  //    if (err < 1e-6) {
  //      std::cout << "Converged with error " << err << std::endl;
  //      return true;
  //    }
  //    else {
  //      std::cout << "Not converged with error " << err << std::endl;
  //      return false;
  //    }
  //}

  bool hasConverged() {
    return num_runs++ > 5;
  }
  void loopUntil() {

    auto outp = taskflow.dump();
    std::ofstream out("full_taskflow.dot");
    out << outp;
    std::cout << "Running taskflow" << std::endl;
    int iter = 0;
    while (!hasConverged()) {
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

      run();
      std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
      std::cout << "Time taken for the iteration: " << std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count() << " milliseconds" << std::endl;
      std::cout << "Iteration " << iter++ << std::endl;
      // updateDoubles(doubles_result);
    }
  }

  void writeDoubles() {
    doubles_result._infer_dimensionality();
    doubles_result._infer_shape();
    doubles_result.write("r2.tns");
  }
};
