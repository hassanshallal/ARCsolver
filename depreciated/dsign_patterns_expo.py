# Summary:
# 1) Decorator pattern can allow you to iterative solueions to solving the problem graph in
# an OOP manner. Decorator is structural for dynamic runtime composition
# 2) Interpreter pattern is behavioral for advanced grammer and DSL.
# 3) State: behavioral adjust behavior depending on internal state
# 4) chain of responsibility: behavioral allows for dynamically adjutsable number/type of handlers
# 5) Observer: very important to automate updating observers to observable
# 6) Strategy: gives the class options t use different implementations of routes
# an artificial cognition must be able to strategize
# 7) Momento can provided an OOP way to go back, you'll need to delegate control to the
# client so as to determine how far it should go back?
# 8) Template solves the problem of several classes sharing some methods and not sharing
# other methods



# The decorator pattern can allow you to iterative solueions to solving the problem graph in
# an OOP manner. Decorator is structural for dynamic runtime composition
class WindowInterface:
    def build(self): pass


class AbstractWindowDecorator(WindowInterface):
    """
    Maintain a reference to a Window object and define an interface
    that conforms to Window's interface.
    """

    def __init__(self, window):
        self._window = window

    def build(self): pass


class Window(WindowInterface):
    def build(self):
        print("Building window")


class BorderDecorator(AbstractWindowDecorator):
    def add_border(self):
        print("Adding border")

    def build(self):
        self.add_border()
        self._window.build()


class VerticalSBDecorator(AbstractWindowDecorator):
    def add_vertical_scroll_bar(self):
        print("Adding vertical scroll bar")

    def build(self):
        self.add_vertical_scroll_bar()
        self._window.build()


class HorizontalSBDecorator(AbstractWindowDecorator):
    def add_horizontal_scroll_bar(self):
        print("Adding horizontal scroll bar")

    def build(self):
        self.add_horizontal_scroll_bar()
        self._window.build()

# The interpreter pattern is behavioral for advanced grammer and DSL.

class Expression:
    def interpret(self, problem_graph):
        pass

class TerminalExpression(Expression):
    def __init__(self, word):
        self.word = word

    def interpret(self, text):
        words = text.split()
        if self.word in text:
            return True
        else:
            return False


class OrExpression(Expression):
    def __init__(self, exp1, exp2):
        self.exp1 = exp1
        self.exp2 = exp2

    def interpret(self, text):
        return self.exp1.interpret(text) or self.exp2.interpret(text)


class AndExpression(Expression):
    def __init__(self, exp1, exp2):
        self.exp1 = exp1
        self.exp2 = exp2

    def interpret(self, text):
        return self.exp1.interpret(text) and self.exp2.interpret(text)


john = TerminalExpression('John')
henry = TerminalExpression('Henry')
mary = TerminalExpression('Mary')
sarah = TerminalExpression('Sarah')
# construct the rule sarah and (mary or (john and henry))


# State pattern just allow handling multiple states for compiling complicated queries from input to output
# Base state class
class ComputerState:
    name = "state"
    allowed = []

    def switch(self, state):
        """ Switch to new state """
        if state.name in self.allowed:
            print ('Current:', self ,' => switched to new state', state.name)
            self.__class__ = state
        else:
            print('Current:', self ,' => switching to', state.name, 'not possible.')

    def __str__(self):
        return self.name

class Off(ComputerState):
    name = "off"
    allowed = ['on']

class On(ComputerState):
    name = "on"
    allowed = ['off','suspend','hibernate']

class Suspend(ComputerState):
    name = "suspend"
    allowed = ['on']

class Hibernate(ComputerState):
    name = "hibernate"
    allowed = ['on']

class Computer:
    def __init__(self):
        self.state = Off()

    def change(self, state):
        self.state.switch(state)

# chain of responsibility: there is some sort of heirarchy, go back to the implementation
class Car:
    def __init__(self, name, water, fuel, oil):
        self.name = name
        self.water = water
        self.fuel = fuel
        self.oil = oil

    def is_fine(self):
        if self.water >= 20 and self.fuel >= 5 and self.oil >= 10:
            print('Car is good to go')
            return True
        else:
            return False


class Handler:
    def __init__(self, successor=None):
        self._successor = successor

    def handle_request(self, car):
        if not car.is_fine() and self._successor is not None:
            self._successor.handle_request(car)


class WaterHandler(Handler):

    def handle_request(self, car):
        if car.water < 20:
            car.water = 100
            print('Added water')
        super().handle_request(car)


class FuelHandler(Handler):

    def handle_request(self, car):
        if car.fuel < 5:
            car.fuel = 100
            print('Added fuel')
        super().handle_request(car)


class OilHandler(Handler):

    def handle_request(self, car):
        if car.oil < 10:
            car.oil = 100
            print('Added oil')
        super().handle_request(car)


# Observer:
class Observable:

    def __init__(self):
        self.observers = []

    def register(self, observer):
        if not observer in self.observers:
            self.observers.append(observer)

    def unregister(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def unregister_all(self):
        if self.observers:
            del self.observers[:]

    def update_observers(self, *args, **kwargs):
        for observer in self.observers:
            observer.update(*args, **kwargs)


class Observer:
    def update(self, *args, **kwargs):
        pass


class AmericanStockMarket(Observer):
    def update(self, *args, **kwargs):
        print("American stock market received: {0}\n{1}".format(args, kwargs))


class EuropeanStockMarket(Observer):
    def update(self, *args, **kwargs):
        print("European stock market received: {0}\n{1}".format(args, kwargs))



# strategy:
class PrimeFinder:

    def __init__(self):
        self.primes = []

    def calculate(self, limit):
        """ Will calculate all the primes below limit. """
        pass

    def out(self):
        """ Prints the list of primes prefixed with which algorithm made it """
        print(self.__class__.__name__)
        for prime in self.primes:
            print(prime)


class HardCodedPrimeFinder(PrimeFinder):
    def calculate(self, limit):
        hardcoded_primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]
        primes = []
        for prime in hardcoded_primes:
            if (prime < limit):
                self.primes.append(prime)


class StandardPrimeFinder(PrimeFinder):
    def calculate(self, limit):
        self.primes = [2]
        # check only odd numbers.
        for number in range(3, limit, 2):
            is_prime = True
            for prime in self.primes:
                if (number % prime == 0):
                    is_prime = False
                    break
            if (is_prime):
                self.primes.append(number)


class PrimeFinderClient:

    def __init__(self, limit):
        self.limit = limit
        if limit <= 50:
            self.finder = HardCodedPrimeFinder()
        else:
            self.finder = StandardPrimeFinder()

    def get_primes(self):
        self.finder.calculate(self.limit)
        self.finder.out()


# Template:
class MakeMeal:

    def buy_ingredients(self, money):
        if money < self.cost:
            assert 0, 'Not enough money to buy ingredients!'

    def prepare(self):
        pass

    def cook(self):
        pass

    def go(self, money):
        self.buy_ingredients(money)
        self.prepare()
        self.cook()

class MakePizza(MakeMeal):
    def __init__(self):
        self.cost = 3

    def prepare(self):
        print("Prepare Pizza - make the dough and add toppings")

    def cook(self):
        print("Cook Pizza - cook in the oven on gas mark 8 for 10 minutes")

class MakeCake(MakeMeal):
    def __init__(self):
        self.cost = 2

    def prepare(self):
        print("Prepare Cake - mix ingredients together and pour into a cake tin")

    def cook(self):
        print("Cook Cake - bake in the oven on gas mark 6 for 20 minutes")
