import Foundation

class Manager {
	
	lazy var employee = Employee("Test Employee", manager: self)
	let name: String
	
	init(_ name: String) {
		self.name = name
	}
	
	deinit {
		print("Manager deinited")
	}
}

class Employee {
	let name: String
	let manager: Manager
	
	init(_ name: String, manager: Manager) {
		self.name = name
		self.manager = manager
	}
	
	deinit {
		print("Employee deinited")
	}
}

public func testManagerEmployee() {
	let manager = Manager("Test Manager")
	print(manager.name)
}
