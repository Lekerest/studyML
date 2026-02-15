def arithmetic(operation):
    if operation == '+':
        return lambda x, y: x + y

    elif operation == '-':
        return lambda x, y: x - y

    elif operation == '*':
        return lambda x, y: x * y

    elif operation == '/':
        return lambda x, y: x / y if y != 0 else 0

    elif operation == '**':
        return lambda x, y: x ** y
    else:
        raise ValueError('Invalid operation')

function = arithmetic('+')
print(function(5, 1))

function = arithmetic('**')
print(function(5, 5))