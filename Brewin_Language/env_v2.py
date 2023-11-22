# The EnvironmentManager class keeps a mapping between each variable name (aka symbol)
# in a brewin program and the Value object, which stores a type, and a value.
from type_valuev2 import Type, Value, create_value, get_printable
class EnvironmentManager:
    def __init__(self):
        self.environment = [{}]
        self.level = 0
        self.parent =None

    # returns a VariableDef object
    def get(self, symbol):
        for env in reversed(self.environment):
            if symbol in env:
                return env[symbol]

        return None

    def set(self, symbol, value):
        for env in reversed(self.environment):
            if symbol in env:
                if env[symbol].type()  == Type.Ref:
                    
                    name= env[symbol].value()[0]
                    env = env[symbol].value()[1]
                    for e in reversed(env):
                        if name in e:
                            e[name] = value
                    return
                else:
                    env[symbol] = value
                return

        # symbol not found anywhere in the environment
        self.environment[-1][symbol] = value

    # create a new symbol in the top-most environment, regardless of whether that symbol exists
    # in a lower environment
    def create(self, symbol, value):
        self.environment[-1][symbol] = value

    # used when we enter a nested block to create a new environment for that block
    def push(self):
        self.level +=1
        self.environment.append({})  # [{}] -> [{}, {}]
        

    # used when we exit a nested block to discard the environment for that block
    def pop(self):
        self.level-=1
        self.environment.pop()
        
