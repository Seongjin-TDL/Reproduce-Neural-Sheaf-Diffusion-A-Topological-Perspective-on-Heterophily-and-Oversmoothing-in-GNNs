
# Day 2: [Python Control Flow, Functions & Git]

## Goals: 
- [x] **Task 1:** Learn about Python functions: defining functions, arguments (positional, keyword), return values, variable scope.
- [x] **Task 2:** Deeper dive into Python data structures: Lists (methods like `.append()`, `.pop()`), Tuples, Dictionaries (accessing items, methods like `.keys()`, `.values()`, `.items()`), Sets.
- [ ] **Task 3:** Install Git on your system. Learn basic Git commands: `git clone <repo_url>`, `git status`, `git add <file>`, `git commit -m "Your message"`, `git push`. Practice by cloning your GitHub repository, making changes locally (e.g., adding a new script), committing, and pushing.     
	- _Resource:_ [Atlassian Git Tutorial (Beginner)](https://www.atlassian.com/git/tutorials/what-is-version-control).
## Topics Covered:

* Learn functions, arguments, return, variable scope.
* Learn list, tuple, dictionary, set.
* Set up Git, Learn basic Git commands


## Key Concepts and Code Experiments:

### Task 1 : Learn about Python functions: defining functions, arguments (positional, keyword), return values, variable scope.

#### Experiment 1 : [Define and call a simple function]


```title:Python
def greet():
    print("Hello from my function!")

greet()
```

```title:Text
Hello from my function!
```

**Greet function is defined with no argument.** 

```title:Python
greet(1)
```

```title:Text
TypeError: greet() takes 0 positional arguments but 1 was given
```

**An error occurs if the type of the input data does not match the expected type of the function argument.**

#### Experiment 2 : [Positional, Keyword and, Default Arguments]

```title:Python
def introduce(name, age, city):
    print(f"{name}, {age}, lives in {city}.")

introduce("Eve", 27, city="Seoul")  # positional, positional, keyword
```

```title:Text
Eve, 27, lives in Seoul.
```

```title:Python
def introduce(name, age, city):
    print(f"{name}, {age}, lives in {city}.")

introduce(name="Frank", 31, "Busan")  # keyword, positional, keyword
```

```title:Text
SyntaxError: positional argument follows keyword argument
```

**There are two ways to provide arguments to a function: positional and keyword. You can also mix them in a function call, as long as all positional arguments come before any keyword arguments.**

```title:Python
def introduce(name="Guest"):
    print(f"{name}")

introduce()
introduce("Eve")          
```

```title:Text
Guest
Eve
```

**We can assign a default value to a function argument.**

```title:Python
def introduce(name="Guest", age=30):
    print(f"{name}, {age}")

introduce()
introduce("Eve")
introduce("Eve", 27)
```

```title:Text
Guest, 30
Eve, 30
Eve, 27
```

**We can assign several default values to several function arguments.**

```title:Python
def introduce(name="Guest", age, city):
    print(f"{name}, {age}, lives in {city}.")

introduce(27, "Seoul")
introduce("Eve", 27, "Seoul")          
```

```title:Text
SyntaxError: non-default argument follows default argument
```

**All non-default (required) parameters must come before any parameters with default values.**

```title:Python
def introduce(name, age=30, city="Busan"):
    print(f"{name}, {age}, lives in {city}.")

introduce("Eve")
introduce("Eve", 27)
introduce("Eve", 27, "Seoul")
```

```title:Text
Eve, 30, lives in Busan.
Eve, 27, lives in Busan.
Eve, 27, lives in Seoul.
```

**It is allowed to have non-default parameters before default parameters in a function definition.**

```title:Python
def introduce(name, age=30, city="Busan"):
    print(f"{name}, {age}, lives in {city}.")

introduce()
```

```title:Text
TypeError: introduce() missing 1 required positional argument: 'name'
```

**An error occurs if the type of the input data does not match the expected type of the function argument.**

#### Experiment 3 : [Return values]

```title:Python
def add_two_numbers(a, b):
    sum_result = a + b
    return sum_result  # Sends the value of sum_result back

# Call the function and store the result in a variable
result = add_two_numbers(5, 3)
print(result)

another_result = add_two_numbers(10, 20) + 2 # You can use the returned value in expressions
print(another_result)
```

```title:Text
8
32
```

**add_two_numbers receives two numbers as input and returns their sum as output.**

```title:Python
def add_two_numbers(a, b):
    sum_result = a + b
    return sum_result

def add1_two_numbers(a,b):
	sum_result = add_two_numbers(a,b)+2
	return sum_result
	
result= add_two_numbers(5,3)
result1= add1_two_numbers(5,3)

print(result)
print(result1)
```

```title:Text
8
10
```

**Previously defined functions can be used inside new function definitions.**

#### Experiment 4 : [Variable Scopes]

```title:Python
def my_local_scope_function():
    message = "I am local to this function." # 'message' is a local variable
    print(message)

my_local_scope_function()
```

```title:Text
I am local to this function.
```

```title:Python
def my_local_scope_function():
    message = "I am local to this function." # 'message' is a local variable
    print(message)
    
print(message)
```

```Title:Text
NameError: name 'message' is not defined
```

**Local variables only exist within the function in which they are defined.**

```title:Python
global_var = "I am global."   # global variable

def print_global():
    print(global_var)  

def first_trial_to_modify_global():
    global_var = "I have been changed globally!"
    print(global_var)

def second_trials_to_modify_global():
    global global_var  
    global_var = "I have been changed globally!!"

print_global()
first_trial_to_modify_global()
print_global()
second_trials_to_modify_global()
print_global()             
```

```title:Text
I am global.
I have been changed globally!
I am global.
I have been changed globally!!
```

**If a global variable and a local variable share the same name, assignments within a function will create or modify the local variable, as demonstrated in "first_trial_to_modify_global". However, when the "global" keyword is used inside a function, it allows the function to modify the global variable directly, as shown in "second_trials_to_modify_global".**

### Task 2: Deeper dive into Python data structures: Lists (methods like `.append()`, `.pop()`), Tuples, Dictionaries (accessing items, methods like `.keys()`, `.values()`, `.items()`), Sets.

#### Experiment 1 : Lists

```title:Python
my_list = [1, 2, 3, "apple","coconut", 5.5]

print(my_list)

my_list.append("banana")    # Add "banana" to the end of the list.
print(my_list)

last_item = my_list.pop()   # Return the last item.
print(my_list)              # The last item is removed from the list.
print(last_item)            

my_list.insert(0,"coconut") # Add "coconut" to the 0th position.
print(my_list)
 
my_list.remove("coconut")   # Remove the first occurrence of "coconut".
print(my_list)

my_list.insert(-1, "watermelon")   # Add "watermelon" to the -1th position.
print(my_list)

my_list.remove("watermelon")       # Remove the first occurrence of "watermelon".
print(my_list)

my_list.insert(-2, "watermelon")   # Add "watermelon" to the -2th position.
print(my_list)
```

```title:Text
[1, 2, 3, 'apple', 'coconut', 5.5]
[1, 2, 3, 'apple', 'coconut', 5.5, 'banana']
[1, 2, 3, 'apple', 'coconut', 5.5]
banana
['coconut', 1, 2, 3, 'apple', 'coconut', 5.5]
[1, 2, 3, 'apple', 'coconut', 5.5]
[1, 2, 3, 'apple', 'coconut', 'watermelon', 5.5]
[1, 2, 3, 'apple', 'coconut', 5.5]
[1, 2, 3, 'apple', 'watermelon', 'coconut', 5.5]
```

**A list is a finite, ordered sequence of elements of various data types that can be modified.**

```title:Python
list1 = [1, 2, 3, 4, 5]
list2 = ['a', 'b', 'c', 'd','e']
concatenated = list1 + list2    # Concatenation.
print(concatenated)

result = list1[:2] + list2 + list1[2:]  # Slice list1 into two parts, insert list2 in between.
print(result)  

list1.extend(list2) # Append elements of list2 to list1, directly modifies list1.
print(list1)
print(list2)
```

```title:Text
[1, 2, 3, 4, 5, 'a', 'b', 'c', 'd', 'e']
[1, 2, 'a', 'b', 'c', 'd', 'e', 3, 4, 5]
[1, 2, 3, 4, 5, 'a', 'b', 'c', 'd', 'e']
['a', 'b', 'c', 'd', 'e']
```

**Concatenation and list slicing create new lists without modifying the originals, while the ".extend()" method modifies the list directly.**

#### Experiment 2 : Tuples

```title:Python
tuple1 = (1, 2, 3)
tuple2 = ('a', 'b', 'c')
result = tuple1 + tuple2

print(result)
print(tuple1, tuple2)
```

```title:Text
(1, 2, 3, 'a', 'b', 'c')
(1, 2, 3) ('a', 'b', 'c')
```

**Concatenation creates new tuples without modifying the originals.**

#### Experiment 3 : Dictionaries

```title:Python
my_dict = {"name": "Alice", "age": 30, "city": "Seoul"}
print(my_dict["name"])   # Output value corresponding to "name".

my_dict["age"]=31
print(my_dict["age"])    # Update "age".

my_dict.pop("city")      # Delete "city" key.
print(my_dict)

my_dict.update({"city":"Seoul"}) # Add "city":"Seoul"
print(my_dict)

print(my_dict.keys())
print(my_dict.values())
print(my_dict.items())
print(list(my_dict.keys()))
print(list(my_dict.values()))
print(list(my_dict.items()))
```

```title:Text
Alice
31
{'name': 'Alice', 'age': 31}
{'name': 'Alice', 'age': 31, 'city': 'Seoul'}
dict_keys(['name', 'age', 'city'])
dict_values(['Alice', 31, 'Seoul'])
dict_items([('name', 'Alice'), ('age', 31), ('city', 'Seoul')])
['name', 'age', 'city']
['Alice', 31, 'Seoul']
[('name', 'Alice'), ('age', 31), ('city', 'Seoul')]
```

**A dictionary is a mutable collection that stores data as key–value pairs.**

#### Experiment  4 : Sets

```title:Python
a = {1, 2, 3}
b = {3, 4, 5}

print(a | b)  # Union
print(a & b)  # Intersection
```

```title:Text
{1, 2, 3, 4, 5}
{3}
```

**A set is an unordered collection of unique items.**.

### Reflections/Challenges:
- 


### Next Steps: 
* Python modules and packages
* In the official "Neural Sheaf Diffusion" GitHub repository, find the `requirements.txt` file or environment setup instructions (often in the README). Note down key dependencies: Python version, PyTorch, PyTorch Geometric, NumPy, etc. (1-2 hours)
* Try to clone the official repository locally into a separate folder. Attempt to create a _new, separate_ conda environment based on its requirements (e.g., `conda create --name official_sheaf_env --file requirements.txt`). 
    
