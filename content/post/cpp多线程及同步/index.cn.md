---
title: C++多线程及同步
# description: 这是一个副标题
date: 2020-10-09

categories:
    - C++
    - 多线程
---


1. `lock_guard<mutex> lock(my_mutex) `
它将尝试获取提供给它的互斥锁的所有权。当控制流离开lock_guard对象的作用域时，lock_guard析构并释放互斥量;

2. `unique_lock <mutex> lock(my_mutex)`，是 lock_guard 的升级加强版，它具有 lock_guard 的所有功能，同时又具有其他很多方法，可以随时加锁解锁，能够应对更复杂的锁定需要；条件变量需要该类型的锁作为参数时必须使用unique_lock

3. 条件变量（condition_variable）唤醒阻塞线程的方式：
notify_all()：唤醒了等待的所有线程，但是这些被唤醒的线程需要去竞争锁，获取锁之后才能执行
notify_one()：唤醒的顺序是阻塞的顺序，先阻塞先唤醒，没有竞争

- 示例1
```cpp
//mutex:互斥量，互斥量:为协调共同对一个共享资源的单独访问而设计的
//condition_variable:、信号量(条件变量),为控制一个具有 有限数量用户 的资源而设计
//condition_variable一般和互斥量一起使用
mutex mtx; //手动上锁、解锁，lock_guard 和 unique_lock 是自动上锁和解锁
mtx.lock();
//代码段
...直接使用mutex的加锁功能，能实现简单的互斥，
...mtx被其他线程lock之后，当前线程会阻塞，直到其他线程unlock，所有的阻塞进程会竞争该锁
...
mtx.unlock()
```

- 示例2
```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <atomic>

using namespace std;

//是否让线程循环的全局变量，使用原子类型，保证多线程下，同一时刻只有一个线程操作该变量
//volatile让线程每次需要存储或读取这个变量的时候，都会直接从变量地址中读取数据
//而不是从编译器优化后的寄存器中读取，防止出现不一致的情况
volatile atomic_bool loop(true); 

class MsgList
{
private:
    deque<int> msg_queue;
    mutex mtx;
    condition_variable cond_var;
public:
    void notify_all()
    {
        cond_var.notify_all();
    }
    void write_msg()
    {
        unique_lock<mutex> lock(mtx); //锁的作用域：整个函数，函数返回后自动释放
        for (size_t i = 0; i < 5; ++i)
        {
            msg_queue.push_back(i);
        }
        //全部数据写入后，唤醒一个阻塞的线程
        //notify_all()唤醒了等待的所有线程，但是这些被唤醒的线程需要去竞争锁，获取锁之后才能执行
        //这里用notify_one()，唤醒的顺序是阻塞的顺序，先阻塞先唤醒，后阻塞后唤醒，没有竞争
        cond_var.notify_one();
    }

    /*
        lock_guard对象时，它将尝试获取提供给它的互斥锁的所有权。
        当控制流离开lock_guard对象的作用域时，lock_guard析构并释放互斥量;
        
        unique_lock 是 lock_guard 的升级加强版，它具有 lock_guard 的所有功能，
        同时又具有其他很多方法，可以随时加锁解锁，能够应对更复杂的锁定需要;
        condition_variable 条件变量需要该类型的锁作为参数，必须使用unique_lock
    */
    void read_msg()
    {
        /*
            在这个例子中，如果不用condition_variable，下面的代码是等价的：
            void read_msg(){
                while (loop){ //死循环
                    unique_lock<mutex> lock(mtx);  //加锁
                    while (!msg_queue.empty()){ //不为空
                        //do something
                    }
                }
            }
            这种写法存在的问题是：
            当写线程没有数据的写入的时候（队列始终为空），所有的读线程会一直循环加锁、检测队列是否为空，造成CPU资源浪费
            所以就有了下面的条件变量
        */
        while (loop)
        {
            unique_lock<mutex> lock(mtx); //实现对{}内的代码段实现能自动上锁和自动解锁，look是自定义的变量名称，在离开作用域时会析构（既是解锁）
            cond_var.wait(lock, [this]() -> bool { return !msg_queue.empty() || !loop; } );
            //调用wait函数：
            //1.如果只有第一个参数，先解锁lock，然后将当前线程添加到当前condition_variable对象的等待线程列表中
            //当前线程将继续阻塞，直到另外某个线程调用 notify_* 唤醒了当前线程，wait函数自动重新上锁并返回，继续执行下面的代码
            
            //2.如果有第二个参数（是一个返回值为bool的可调用对象，这里用匿名函数）
            //先解锁、阻塞和1是一样的，当前线程被其他线程唤醒后，再判可调用对象的返回值(bool),如果为fals,
            //如果返回true（这里是队列不为空返回true），wait函数自动重新上锁并返回，继续执行下面的代码
            
            //这里的!loop是判断是否要退出循环的，loop为false， !lopp为true，则会执行下面的break代码，退出死循环，结束线程

            //在第二个参数中，还可以判断读线程的个数，既是允许最大的读线程个数（限制资源的用户量）
            
            //wait函数总结就是，先解锁和阻塞，等唤醒；如果有第二个参数，被唤醒后，再判断第一个参数的返回值来决定继续阻塞，还是wait函数返回并执行下面的代码
            if (!loop) break;
         
            cout << endl<< "thread id: " << this_thread::get_id() << " ; 出队元素:";
            while (!msg_queue.empty()) //将队列元素全部出队
            {
                int tmp = msg_queue.front();
                msg_queue.pop_front();
                cout << tmp  << ' '; 
            }
        }   
    }

    MsgList(/* args */){};
    ~MsgList(){};
};

int main()
{
    MsgList mlist;

    int read_thread_num = 5; //5个读线程
    thread read_threads[read_thread_num];

    for (int i = 0; i < read_thread_num; ++i)       
    {
        read_threads[i] = thread(&MsgList::read_msg, &mlist); 
        //第一个参数是函数指针（函数的地址），第二个参数是函数的参数，这里是一个实例化的对象的地址
        //让这个线程去执行这个实例化的对象的函数
    }

    int write_thread_num = 50; //50个写线程，一次性的
    thread write_threads[write_thread_num];
    for (int i = 0; i < write_thread_num; ++i)
    {
        write_threads[i] = thread(&MsgList::write_msg, &mlist);
        // write_threads[i].join();
        this_thread::sleep_for(chrono::milliseconds(20)); //主线程休眠20ms
    }

    for (int i = 0; i < write_thread_num; ++i)  write_threads[i].join(); //等待所有的写线程结束
    
    loop = false; //设置loop为false， 退出循环
    mlist.notify_all(); //唤醒所有的阻塞的线程，退出循环,结束线程

    for (int i = 0; i < read_thread_num; ++i) read_threads[i].join(); //等待所有的读线程结束

    return 0;
}
```