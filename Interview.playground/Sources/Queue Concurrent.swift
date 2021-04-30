import Foundation

public func testConcurrentQueue() {
	let queue = DispatchQueue(label: "c", qos: .utility, attributes: .concurrent, autoreleaseFrequency: .inherit, target: nil)
	
	queue.async {
		print("c")
		
		queue.async {
			print("d")
		}
		
		queue.async {
			print("e")
		}
		
		queue.sync {
			print("f")
		}
		
		queue.async {
			print("g")
		}
		
		DispatchQueue.main.sync {
			print("h")
		}
	}
}
