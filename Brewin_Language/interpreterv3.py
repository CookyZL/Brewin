import copy
from enum import Enum

from brewparse import parse_program
from env_v2 import EnvironmentManager
from intbase import InterpreterBase, ErrorType
from type_valuev2 import Type, Value, create_value, get_printable


class ExecStatus(Enum):
    CONTINUE = 1
    RETURN = 2


# Main interpreter class
class Interpreter(InterpreterBase):
    # constants
    NIL_VALUE = create_value(InterpreterBase.NIL_DEF)
    TRUE_VALUE = create_value(InterpreterBase.TRUE_DEF)
    BIN_OPS = {"+", "-", "*", "/", "==", "!=", ">", ">=", "<", "<=", "||", "&&"}

    # methods
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)
        self.trace_output = trace_output
        self.__setup_ops()

    # run a program that's provided in a string
    # usese the provided Parser found in brewparse.py to parse the program
    # into an abstract syntax tree (ast)
    def run(self, program):
        ast = parse_program(program)
        self.__set_up_function_table(ast)
        self.id=0
        self.env = EnvironmentManager()
        main_func = self.__get_func_by_name("main", 0)
        self.__run_statements(main_func.get("statements"))

    def __set_up_function_table(self, ast):
        self.func_name_to_ast = {}
        for func_def in ast.get("functions"):
            func_name = func_def.get("name")
            num_params = len(func_def.get("args"))
            if func_name not in self.func_name_to_ast:
                self.func_name_to_ast[func_name] = {}
            self.func_name_to_ast[func_name][num_params] = func_def

    def __get_func_by_name(self, name, num_params):
        if isinstance(name,tuple):
            name = name[0]
        if name not in self.func_name_to_ast:
            super().error(ErrorType.NAME_ERROR, f"Function {name} not found")
        candidate_funcs = self.func_name_to_ast[name]
        if num_params not in candidate_funcs:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Function {name} taking {num_params} params not found",
            )
        return candidate_funcs[num_params]

    def __run_statements(self, statements):
        self.env.push()
        for statement in statements:
            if self.trace_output:
                print(statement)
            status = ExecStatus.CONTINUE
            if statement.elem_type == InterpreterBase.FCALL_DEF:
                self.__call_func(statement)
            elif statement.elem_type == "=":
                self.__assign(statement)
            elif statement.elem_type == InterpreterBase.RETURN_DEF:
                status, return_val = self.__do_return(statement)
                print(return_val.type())
                if return_val.type() == Type.Func:
                    return_val= Value(return_val.type(), (return_val.value(), self.id))
                    self.id+=1
            elif statement.elem_type == Interpreter.IF_DEF:
                status, return_val = self.__do_if(statement)
            elif statement.elem_type == Interpreter.WHILE_DEF:
                status, return_val = self.__do_while(statement)

            if status == ExecStatus.RETURN:
                self.env.pop()
                var_name = return_val.value()
                if not isinstance(var_name, dict) and var_name in self.func_name_to_ast:
                    if len(self.func_name_to_ast[var_name])>1:
                        super().error(
                            ErrorType.NAME_ERROR, "Multiple function with same name"
                        )
                return (status, return_val)

        self.env.pop()
        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __lambda_func(self, func_name, call_node):
        func_ast = func_name.value()
        actual_args = call_node.get("args")
        formal_args = func_ast.get("args")
        if len(actual_args) != len(formal_args):
            super().error(
                ErrorType.TYPE_ERROR,
                f"Function {func_ast.get('name')} with {len(actual_args)} args not found",
            )
        level = self.env.level
        tmp_env = self.env
        ref_val = []
        self.env= func_ast.get("environment")
        self.env.parent = tmp_env
        for formal_ast, actual_ast in zip(formal_args, actual_args):
            if formal_ast.elem_type == "refarg":

                arg_name = formal_ast.get("name")
                var_name = actual_ast.get('name')
                ref_val.append(var_name)
                if self.env.get(var_name) == None:
                    val = tmp_env.get(var_name)
                    env = tmp_env.environment[:level+1]
                else:
                    val = self.env.get(var_name)
                    env = self.env.environment[:level+1]
    
                if val.type() == Type.Ref:
                    var_name = val.value()[0]
                    env = val.value()[1]
                
                self.env.create(arg_name, Value(Type.Ref, [var_name, env]))
            else:
                result = copy.deepcopy(self.__eval_expr(actual_ast))
                arg_name = formal_ast.get("name")
                self.env.create(arg_name, result)
        _, return_val = self.__run_statements(func_ast.get("statements"))
        for var in ref_val:
            val = self.env.get(var)

            if val == None or val.type() == Type.Ref:
                continue
            tmp_env.set(var,val)
        self.env = tmp_env
        self.env.parent = None
        
        return return_val
    

    def __call_func(self, call_node):
        func_name = call_node.get("name")
        if func_name == "print":
            return self.__call_print(call_node)
        if func_name == "inputi":
            return self.__call_input(call_node)
        if func_name == "inputs":
            return self.__call_input(call_node)
        if self.env.get(func_name):
            func_name = self.env.get(func_name)
            if func_name.type() == Type.Func:
                func_name = func_name.value()
            elif func_name.type() == Type.Lambda:
                return self.__lambda_func(func_name, call_node)
            elif func_name.type() == Type.Ref:
                name= func_name.value()[0]
                env = func_name.value()[1]
        
                for e in reversed(env):
                    if name in e:
                        val = e[name]
                if val.type() == Type.Lambda:
                    return self.__lambda_func(val, call_node)
            else:
                super().error(
                        ErrorType.TYPE_ERROR,
                        f"This is not a function!",
                    )

        actual_args = call_node.get("args")
        func_ast = self.__get_func_by_name(func_name, len(actual_args))
        formal_args = func_ast.get("args")
        if len(actual_args) != len(formal_args):
            super().error(
                ErrorType.NAME_ERROR,
                f"Function {func_ast.get('name')} with {len(actual_args)} args not found",
            )
        level = self.env.level
        self.env.push()
        for formal_ast, actual_ast in zip(formal_args, actual_args):
            if formal_ast.elem_type == "refarg":
                # print("here ref")
                # print(call_node)
                
                arg_name = formal_ast.get("name")
                var_name = actual_ast.get('name')
                val = self.env.get(var_name)
                env = self.env.environment[:level+1]
                if val.type() == Type.Ref:
                    var_name = val.value()[0]
                    env = val.value()[1]
                
                self.env.create(arg_name, Value(Type.Ref, [var_name, env]))
            else:
                result = self.__eval_expr(actual_ast)
                # if result.type()!=Type.Lambda:
                result = copy.deepcopy(result)
                if not isinstance(result.value(), dict):
                    if result.value() in self.func_name_to_ast:
                        if len(self.func_name_to_ast[result.value()])>1:
                            super().error(
                                ErrorType.NAME_ERROR, "Multiple function with same name"
                            )
                arg_name = formal_ast.get("name")
                self.env.create(arg_name, result)
        _, return_val = self.__run_statements(func_ast.get("statements"))
        self.env.pop()
        return return_val

    def __call_print(self, call_ast):
        output = ""
        for arg in call_ast.get("args"):
            result = self.__eval_expr(arg)  # result is a Value object
            if result == None:
                result = Interpreter.NIL_VALUE
            output = output + get_printable(result)
        super().output(output)
        return Interpreter.NIL_VALUE

    def __call_input(self, call_ast):
        args = call_ast.get("args")
        if args is not None and len(args) == 1:
            result = self.__eval_expr(args[0])
            if result == None:
                result = Interpreter.NIL_VALUE
            super().output(get_printable(result))
        elif args is not None and len(args) > 1:
            super().error(
                ErrorType.NAME_ERROR, "No inputi() function that takes > 1 parameter"
            )
        inp = super().get_input()
        if call_ast.get("name") == "inputi":
            return Value(Type.INT, int(inp))
        if call_ast.get("name") == "inputs":
            return Value(Type.STRING, inp)

    def __assign(self, assign_ast):
        var_name = assign_ast.get("name")
        expression = assign_ast.get("expression")

        value_obj = self.__eval_expr(expression) 
        value = value_obj.value()
        if isinstance(value,tuple):
            value = value[0]
        if value_obj.type() == Type.Func and len(self.func_name_to_ast[value])>1:
            super().error(
                ErrorType.NAME_ERROR, "Multiple function with same name"
            )
        # if value_obj.type() == Type.Lambda:
        #     value_obj = expression


        self.env.set(var_name, value_obj)

    def __eval_expr(self, expr_ast):
        # print("here expr")
        # print(expr_ast)
        if expr_ast.elem_type == InterpreterBase.NIL_DEF:
            # print("getting as nil")
            return Interpreter.NIL_VALUE
        if expr_ast.elem_type == InterpreterBase.INT_DEF:
            return Value(Type.INT, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.STRING_DEF:
            # print("getting as str")
            return Value(Type.STRING, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.BOOL_DEF:
            return Value(Type.BOOL, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.VAR_DEF:
            var_name = expr_ast.get("name")
            val = self.env.get(var_name)
            if val is None:
                if var_name in self.func_name_to_ast:
                    return Value(Type.Func, (var_name))
                else:
                    if self.env.parent !=None:
                        val = self.env.parent.get(var_name)
                        if val == None:
                            super().error(ErrorType.NAME_ERROR, f"Variable {var_name} not found")

                    else:
                        super().error(ErrorType.NAME_ERROR, f"Variable {var_name} not found")
            if val.type() == Type.Lambda:
                return val
            if val.type() == Type.Ref:
                name= val.value()[0]
                env = val.value()[1]
                # print(name)
                # print(env)
                for e in reversed(env):
                    if name in e:
                        val = e[name]
            return val
        if expr_ast.elem_type == InterpreterBase.FCALL_DEF:
            return self.__call_func(expr_ast)
        if expr_ast.elem_type in Interpreter.BIN_OPS:
            return self.__eval_op(expr_ast)
        if expr_ast.elem_type == Interpreter.NEG_DEF:
            return self.__eval_unary(expr_ast, Type.INT, lambda x: -1 * x)
        if expr_ast.elem_type == Interpreter.NOT_DEF:
            return self.__eval_unary(expr_ast, Type.BOOL, lambda x: not x)
        if expr_ast.elem_type == Interpreter.LAMBDA_DEF:
            return self.__eval_lambda(expr_ast)
        
    def __eval_lambda(self, expr_ast):
        val =  expr_ast.dict
        env = EnvironmentManager()
        env.environment = copy.deepcopy(self.env.environment)
        env.push()
        val["environment"] = env
        return Value(Type.Lambda, val)
        

    def __eval_op(self, arith_ast):
        left_value_obj = self.__eval_expr(arith_ast.get("op1"))
        right_value_obj = self.__eval_expr(arith_ast.get("op2"))


        if left_value_obj.type() != Type.Func and right_value_obj.type() != Type.Func:
            if (left_value_obj.type() == Type.INT or right_value_obj.type() == Type.INT) and arith_ast.elem_type in ['||', '&&', '==', '!=']:
                    if left_value_obj != Type.Func:
                        left_value_obj = Value(Type.BOOL, bool(left_value_obj.value()))
                    if right_value_obj != Type.Func:
                        right_value_obj = Value(Type.BOOL, bool(right_value_obj.value()))
        
        if arith_ast.elem_type in ["+", "-", "*", "/"]:
            if left_value_obj.type() == Type.BOOL:
                if left_value_obj.value() == True:
                    left_value_obj = Value(Type.INT, 1)
                else:
                    left_value_obj = Value(Type.INT, 0)

            if right_value_obj.type() == Type.BOOL:
                if right_value_obj.value() == True:
                    right_value_obj = Value(Type.INT, 1)
                else:
                    right_value_obj = Value(Type.INT, 0)        
            
        if not self.__compatible_types(
            arith_ast.elem_type, left_value_obj, right_value_obj
        ):
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible types for {arith_ast.elem_type} operation",
            )
        if arith_ast.elem_type not in self.op_to_lambda[left_value_obj.type()]:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible operator {arith_ast.elem_type} for type {left_value_obj.type()}",
            )
        f = self.op_to_lambda[left_value_obj.type()][arith_ast.elem_type]
        # print("here eval")
        # print(arith_ast)
        # print("evaluating " + str(left_value_obj.type()) + " " + str(arith_ast.elem_type))
        # print("obj left: " + str(left_value_obj.value()))
        return f(left_value_obj, right_value_obj)

    def __compatible_types(self, oper, obj1, obj2):
        # DOCUMENT: allow comparisons ==/!= of anything against anything
        if oper in ["==", "!="]:
            return True
        return obj1.type() == obj2.type()

    def __eval_unary(self, arith_ast, t, f):
        value_obj = self.__eval_expr(arith_ast.get("op1"))
        if value_obj.type() == Type.INT:
            return Value(Type.BOOL, bool(f(value_obj.value())))
        if value_obj.type() != t:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible type for {arith_ast.elem_type} operation",
            )
        return Value(t, f(value_obj.value()))

    def __setup_ops(self):
        self.op_to_lambda = {}
        # set up operations on integers
        self.op_to_lambda[Type.INT] = {}
        self.op_to_lambda[Type.INT]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.INT]["-"] = lambda x, y: Value(
            x.type(), x.value() - y.value()
        )
        self.op_to_lambda[Type.INT]["*"] = lambda x, y: Value(
            x.type(), x.value() * y.value()
        )
        self.op_to_lambda[Type.INT]["/"] = lambda x, y: Value(
            x.type(), x.value() // y.value()
        )
        self.op_to_lambda[Type.INT]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.INT]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )
        self.op_to_lambda[Type.INT]["<"] = lambda x, y: Value(
            Type.BOOL, x.value() < y.value()
        )
        self.op_to_lambda[Type.INT]["<="] = lambda x, y: Value(
            Type.BOOL, x.value() <= y.value()
        )
        self.op_to_lambda[Type.INT][">"] = lambda x, y: Value(
            Type.BOOL, x.value() > y.value()
        )
        self.op_to_lambda[Type.INT][">="] = lambda x, y: Value(
            Type.BOOL, x.value() >= y.value()
        )
        #  set up operations on strings
        self.op_to_lambda[Type.STRING] = {}
        self.op_to_lambda[Type.STRING]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.STRING]["=="] = lambda x, y: Value(
            Type.BOOL, x.value() == y.value()
        )
        self.op_to_lambda[Type.STRING]["!="] = lambda x, y: Value(
            Type.BOOL, x.value() != y.value()
        )
        #  set up operations on bools
        self.op_to_lambda[Type.BOOL] = {}
        self.op_to_lambda[Type.BOOL]["&&"] = lambda x, y: Value(
            x.type(), x.value() and y.value()
        )
        self.op_to_lambda[Type.BOOL]["||"] = lambda x, y: Value(
            x.type(), x.value() or y.value()
        )
        self.op_to_lambda[Type.BOOL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.BOOL]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

        #  set up operations on nil
        self.op_to_lambda[Type.NIL] = {}
        self.op_to_lambda[Type.NIL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.NIL]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

        #function comparison
        self.op_to_lambda[Type.Func] = {}
        self.op_to_lambda[Type.Func]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.Func]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

        #lambda comparison
        self.op_to_lambda[Type.Lambda] = {}
        self.op_to_lambda[Type.Lambda]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.Lambda]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )


    def __do_if(self, if_ast):
        cond_ast = if_ast.get("condition")
        result = self.__eval_expr(cond_ast)
        if result.type() != Type.BOOL and result.type() != Type.INT:
            super().error(
                ErrorType.TYPE_ERROR,
                "Incompatible type for if condition",
            )
        if result.value():
            statements = if_ast.get("statements")
            status, return_val = self.__run_statements(statements)
            return (status, return_val)
        else:
            else_statements = if_ast.get("else_statements")
            if else_statements is not None:
                status, return_val = self.__run_statements(else_statements)
                return (status, return_val)

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __do_while(self, while_ast):
        cond_ast = while_ast.get("condition")
        run_while = Interpreter.TRUE_VALUE
        while run_while.value():
            run_while = self.__eval_expr(cond_ast)

            if run_while.type() != Type.BOOL and run_while.type() != Type.INT:
                super().error(
                    ErrorType.TYPE_ERROR,
                    "Incompatible type for while condition",
                )
            if run_while.value():
                statements = while_ast.get("statements")
                status, return_val = self.__run_statements(statements)
                if status == ExecStatus.RETURN:
                    return status, return_val

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __do_return(self, return_ast):
        expr_ast = return_ast.get("expression")
        if expr_ast is None:
            return (ExecStatus.RETURN, Interpreter.NIL_VALUE)
        value_obj = copy.deepcopy(self.__eval_expr(expr_ast))
        return (ExecStatus.RETURN, value_obj)
