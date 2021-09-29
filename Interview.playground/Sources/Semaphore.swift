
/*
//https://open.umn.edu/opentextbooks/textbooks/the-little-book-of-semaphores
//https://greenteapress.com/wp/semaphores/
//https://greenteapress.com/semaphores/LittleBookOfSemaphores.pdf
*/

import Dispatch

typealias Semaphore = DispatchSemaphore

class Rendezvous {

    let a1done = DispatchSemaphore(value: 0)
    let b1done = DispatchSemaphore(value: 0)

    func a() {
        print("a1")
        a1done.signal()
        b1done.wait()
        print("a2")
    }

    func b() {
        print("b1")
        b1done.signal()
        a1done.wait()
        print("b2")
    }

    func run() {
        //a1 -> b2
        //b1 -> a2

        DispatchQueue.global().async {
            self.a()
        }

        self.b()
    }
}

extension Semaphore {
    func signal(_ n: Int) {
        for _ in 0..<n {
            signal()
        }
    }
}

class Barrier {

    private let count: Int
    private lazy var pending = count
    private let mutex = Semaphore(value: 1)
    private let turnstile = Semaphore(value: 0)
    private let turnstile2 = Semaphore(value: 0)

    init(_ count: Int) {
        self.count = count
    }

    //Signals arrival at rendezvous and
    //waits for all other threads to signal rendezvous
    func phase1() {
        mutex.wait()
        pending -= 1
        if pending == 0 {
            turnstile.signal(count)
        }
        mutex.signal()
        turnstile.wait()
    }

    //Makes sure next loop starts only
    //when all threads have completed last
    func phase2() {
        mutex.wait()
        pending += 1
        if pending == count {
            turnstile2.signal(count)
        }
        mutex.signal()
        turnstile2.wait()
    }

    func wait() {
        phase1()
        phase2()
    }

    static func test(threads: Int) {
        let barrier = Barrier(threads)

        func execute(_ i: Int) {
            print(i, "at rendezvous")
            barrier.wait()
            print("complete", i)
        }

        for i in 0..<threads {
            DispatchQueue.global().async {
                execute(i)
            }
        }
    }
}

class LightSwitch {

    private var counter = 0
    private let mutex = Semaphore(value: 1)

    func lock(semaphore: Semaphore) {
        mutex.wait()
        counter += 1
        if counter == 1 {
            semaphore.wait()
        }
        mutex.signal()
    }

    func unlock(semaphore: Semaphore) {
        mutex.wait()
        counter -= 1
        if counter == 0 {
            semaphore.signal()
        }
        mutex.signal()
    }
}
