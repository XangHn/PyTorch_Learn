class Person:
    def __call__(self, name):
        print("__call__: " + name)

    def hello(self, name):
        print("Hello " + name)


person = Person()
person("zhangsan")
person.hello("lisi")