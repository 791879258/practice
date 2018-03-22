链接：https://www.nowcoder.com/questionTerminal/4c776177d2c04c2494f2555c9fcc1e49
来源：牛客网

/*
* 1.dataStack为存储数据的栈，minStack为存储最小值的栈；
* 2.push的时候将value值与minStack中的top值比较，小则minStack push value，大则push top值
*/
class Solution {
public:
    stack<int> dataStack, minStack;
    void push(int value) {
        dataStack.push(value);
        if (minStack.empty()) {
            minStack.push(value);
        }
        else{
            int min = minStack.top();
            value<=min?minStack.push(value):minStack.push(min);
        }
         
    }
    void pop() {
        dataStack.pop();
        minStack.pop();
    }
    int top() {
        return  dataStack.top();
    }
    int min() {
        return minStack.top();
    }
};