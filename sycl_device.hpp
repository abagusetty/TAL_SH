#ifndef __SYCL_DEVICE_HPP__
#define __SYCL_DEVICE_HPP__

#include <CL/sycl.hpp>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>

namespace talsh {

auto exception_handler = [](cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (cl::sycl::exception const &e) {
      std::cerr << "Caught asynchronous SYCL exception: " << e.what() << std::endl
                << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    }
  }
};

/// device extension
class device_ext : public cl::sycl::device {
public:
  device_ext() : cl::sycl::device() {}
  ~device_ext() {
    for (auto q : _queues) {
      delete q;
      q = nullptr;
    }
    for (auto e : _events) {
      delete e;
      e = nullptr;
    }
  }
  device_ext(const cl::sycl::device &base) : cl::sycl::device(base) {
    cl::sycl::queue *_dev_queue = new cl::sycl::queue(base, exception_handler,
						      cl::sycl::property::queue::in_order());
    _queues.insert(_dev_queue);
    _active_queue = _dev_queue;
  }

  void reset() {
    for (auto q : _queues) {
      delete q;
      q = nullptr;
    }
    _queues.clear();
    for (auto e : _events) {
      delete e;
      e = nullptr;
    }
    _events.clear();    
  }

  void queues_wait_and_throw() {
    for (auto q : _queues) {
      q->wait_and_throw();
    }
  }
  cl::sycl::queue &create_queue() {
    cl::sycl::queue *queue = new cl::sycl::queue(_active_queue->get_context(), _active_queue->get_device(),
						 exception_handler, cl::sycl::property::queue::in_order());
    _queues.insert(queue);
    _active_queue = queue;
    return *queue;
  }
  cl::sycl::event &create_event() {
    cl::sycl::event *event = new cl::sycl::event();
    _events.insert(event);
    return *event;
  }
  void destroy_queue(cl::sycl::queue *&queue) {
    _queues.erase(queue);
    delete queue;
    queue = nullptr;
  }
  void destroy_event(cl::sycl::event *&event) {
    _events.erase(event);
    delete event;
    event = nullptr;
  }
  void set_active_queue() {
    _active_queue = *(_queues.end());
  }
  cl::sycl::queue &get_active_queue() {
    return *_active_queue;
  }

private:
  cl::sycl::queue *_active_queue;
  std::set<cl::sycl::queue*> _queues;
  std::set<cl::sycl::event*> _events;
};

/// device manager
class dev_mgr {
public:
  device_ext &current_device() {
    unsigned int dev_id=current_device_id();
    check_id(dev_id);
    return *_devs[dev_id];
  }
  device_ext &get_device(unsigned int id) const {
    check_id(id);
    return *_devs[id];
  }
  unsigned int current_device_id() const {
    return _currentActiveDevice;
  }
  void select_device(unsigned int id) {
    check_id(id);
    _currentActiveDevice=id;
    _devs[id]->set_active_queue();
  }
  unsigned int device_count() { return _devs.size(); }

  /// Returns the instance of device manager singleton.
  static dev_mgr &instance() {
    static dev_mgr d_m;
    return d_m;
  }
  dev_mgr(const dev_mgr &) = delete;
  dev_mgr &operator=(const dev_mgr &) = delete;
  dev_mgr(dev_mgr &&) = delete;
  dev_mgr &operator=(dev_mgr &&) = delete;

private:
  dev_mgr() {
    std::vector<cl::sycl::device> sycl_all_devs =
      cl::sycl::device::get_devices(cl::sycl::info::device_type::gpu);
    for (auto &dev : sycl_all_devs) {
      _devs.push_back(std::make_shared<device_ext>(dev));
    }
  }
  void check_id(unsigned int id) const {
    if (id >= _devs.size()) {
      throw std::runtime_error("invalid device id");
    }
  }
  std::vector<std::shared_ptr<device_ext>> _devs;
  unsigned int _currentActiveDevice;
};

/// Util function to get the defualt queue of current device
static inline cl::sycl::queue &get_queue() {
  return dev_mgr::instance().current_device().get_active_queue();
}

  /// Util function to get the total number of GPUs on a node
static inline void get_device_count(int* gpus_per_node) {
  *gpus_per_node = dev_mgr::instance().device_count();
}

/// Util function to get the current device.
static inline device_ext &get_current_device() {
  return dev_mgr::instance().current_device();
}

/// Util function to set, get a device by id.
static inline device_ext &get_device(unsigned int id) {
  return dev_mgr::instance().get_device(id);
}
static inline void set_device(unsigned int gpu_id) {
  return dev_mgr::instance().select_device(gpu_id);
}


} // namespace talsh

#endif // __SYCL_DEVICE_HPP__
