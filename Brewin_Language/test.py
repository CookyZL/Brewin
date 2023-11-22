from interpreterv3 import Interpreter

program = """
func bar(a) {
    b = 1;
    
    f = lambda(x) {
        print(x);
    };
    
    return lambda(c) {
        return f(c);
    };
}

func main() {
    a = bar(1);
    b = a(2);
}

"""


interpreter = Interpreter()
interpreter.run(program)