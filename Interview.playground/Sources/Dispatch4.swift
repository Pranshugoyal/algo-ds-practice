protocol Fighter {
    func fight()
}

extension Fighter {
    func fight() {
        punch()
    }
    
    func punch() {
        print("Fighter punch")
    }
}

protocol MagicalFighter: Fighter {
    func castSpell()
}

extension MagicalFighter {
    func castSpell() {
        print("MagicalFighter casted a spell")
    }
    
    func punch() {
        castSpell()
        print("Magical fighter punch")
    }
}

class Hero: MagicalFighter {
    func castSpell() {
        print("Hero casted special spell")
    }
    
    func punch() {
        print("Hero punch")
    }
}

public func testDispatch4() {
    let gordo: Fighter = Hero()
    gordo.fight()
}
