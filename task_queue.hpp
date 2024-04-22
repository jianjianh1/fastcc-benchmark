#include "contract.hpp"
#include <taskflow/taskflow.hpp>

class TaskQueue {
  tf::Taskflow taskflow;
  //tf::Executor executor{1};
  tf::Executor executor;
  int task_id = 0;

public:
  void run() {
    auto outp = taskflow.dump();
    std::ofstream out("full_taskflow.dot");
    out << outp;
    std::cout << "Running taskflow" << std::endl;

    executor.run(taskflow).wait();
  }
  template <class Res, class L, class R>
  void addContraction(Tensor<Res> &result, Tensor<L> left, Tensor<R> right,
                      CoOrdinate left_contr, CoOrdinate left_batch,
                      CoOrdinate right_contr, CoOrdinate right_batch) {
    // TODO potentially we can make it faster using the Symbolic result, but
    // that's for later.
    taskflow.emplace([&result, left, right, left_contr, left_batch, right_contr,
                      right_batch]() mutable {
      result.fill_values(left, right, left_contr, left_batch, right_contr,
                         right_batch);
      std::cout << result.get_nonzeros().size() << std::endl;
    });
  }
  template <class Res, class L, class R>
  tf::Task makeTask(Tensor<Res> &result, Tensor<L> &left, Tensor<R> &right,
                    CoOrdinate left_contr, CoOrdinate left_batch,
                    CoOrdinate right_contr, CoOrdinate right_batch,
                    std::string lname, std::string rname) {
    // TODO potentially we can make it faster using the Symbolic result, but
    // that's for later.
    //std::cout << "Queue has " << taskflow.num_tasks() << " tasks" << std::endl;
    auto task_id = this->task_id++;
    return taskflow
        .emplace([&result, &left, &right, left_contr, left_batch, right_contr,
                  right_batch, task_id, lname, rname]() mutable {
          //std::cout << "Task " << task_id << " started" << std::endl;
          //std::cout << "Left " << lname << " has " << left.get_nonzeros().size()
          //          << std::endl;
          //std::cout << "Right " << rname << " has "
          //          << right.get_nonzeros().size() << std::endl;
          result.fill_values(left, right, left_contr, left_batch, right_contr,
                             right_batch);
          //std::cout << "Result is " << result.get_dimensionality() << "D"
          //          << std::endl;
          //std::cout << "Result has " << result.get_nonzeros().size()
          //          << std::endl;
        })
        .name(lname + " * " + rname);
  }
};
