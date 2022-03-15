## 习题

### LRU LFU

> [!NOTE] **[LeetCode 146. LRU缓存机制](https://leetcode-cn.com/problems/lru-cache/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准**

```cpp
// 归一后的写法
class LRUCache {
public:
    struct Node {
        int k, v;
    };

    int cap;
    list<Node> cache;
    unordered_map<int, list<Node>::iterator> hash;

    LRUCache(int capacity) {
        this->cap = capacity;
    }
    
    Node remove(list<Node>::iterator it) {
        auto [k, v] = *it;
        cache.erase(it);
        hash.erase(k);
        return {k, v};
    }

    void insert(int k, int v) {
        cache.push_front({k, v});
        hash[k] = cache.begin();
    }

    int get(int key) {
        if (hash.find(key) == hash.end())
            return -1;
        auto it = hash[key];
        auto [k, v] = remove(it);
        insert(k, v);
        return v;
    }
    
    void put(int key, int value) {
        if (hash.find(key) == hash.end()) {
            if (hash.size() == cap) {
                auto _ = remove( -- cache.end());
            }
        } else
            auto _ = remove(hash[key]);
        insert(key, value);
    }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
```

##### **C++ 裸链表**

```cpp
class LRUCache {
public:
    struct Node {
        int key, val;
        Node *left, *right;
        Node(int _key, int _val): key(_key), val(_val), left(NULL), right(NULL) {}
    }*L, *R;
    unordered_map<int, Node*> hash;
    int n;

    void remove(Node* p) {
        p->right->left = p->left;
        p->left->right = p->right;
    }

    void insert(Node* p) {
        p->right = L->right;
        p->left = L;
        L->right->left = p;
        L->right = p;
    }

    LRUCache(int capacity) {
        n = capacity;
        L = new Node(-1, -1), R = new Node(-1, -1);
        L->right = R, R->left = L;
    }

    int get(int key) {
        if (hash.count(key) == 0) return -1;
        auto p = hash[key];
        remove(p);
        insert(p);
        return p->val;
    }

    void put(int key, int value) {
        if (hash.count(key)) {
            auto p = hash[key];
            p->val = value;
            remove(p);
            insert(p);
        } else {
            if (hash.size() == n) {
                auto p = R->left;
                remove(p);
                hash.erase(p->key);
                delete p;
            }
            auto p = new Node(key, value);
            hash[key] = p;
            insert(p);
        }
    }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
```

##### **Python**

```python
# 这道题面试官就是希望我们实现一个简单的双向链表（删除和插入都是O(1))
# python有一种结合哈希表与双向链表的数据结构OrderedDict，第二种方法里将写一下。


# 法一：双向链表 + 哈希表 O(1)
# 1. k-v的增删改查显然需要一个哈希表来实现，这道题困难之处在于删除时，如果查找删除的元素
# 2. 借用一个双向链表来实现，初始时，链表有两个dummy节点，分别为 L 和 R；新插入一个元素，将其插入到 L 的后面，然后在哈希表中记录新元素的结点地址；
# 3. 遇到要删除的时候， 删除 R 前面的那个节点，同时释放哈希表的内存
# 4. 遇到 get 操作或者 put 操作时，通过哈希表找到节点的地址，然后将其取出，放到 L 的后main，然后修改哈希表。
# 总之，越新越近的元素放在链表头的位置；越老越久的元素放在链表尾的位置，remove出元素的时候，就删除链表尾的元素

class Node:  # 初始化双向链表的数据结构
    def __init__(self, key, val):
        self.key = key
        self.val = val 
        self.left = None    
        self.right = None 

class LRUCache:
    def __init__(self, capacity: int):  # 初始化需要用的哈希表，以及双链表的头尾节点
        self.my_dict = collections.defaultdict(int)  # 哈希表里存的是 node.key -- node!!!
        self.L = Node(-1, -1)
        self.R = Node(-1, -1)
        self.L.right = self.R   # 将头尾节点的左右指针初始化
        self.R.left = self.L 
        self.maxLen = capacity  # 最大容量值获取出来

    def get(self, key: int) -> int:
        if key in self.my_dict:  # 如果key已经在哈希表中了，那就从哈希表中先找到对应的节点
            p = self.my_dict[key]
            self.remove(p)  # 将其删除掉
            self.insert(p)  # 然后再插入到链表中（这个时候会插入到链表头部的位置，代表此刻最新：更新位置）
            return p.val  
        else:
            return -1  
        
    def put(self, key: int, value: int) -> None:  # 放入到缓存中
        if key in self.my_dict:    # 如果key已经在缓存中了，那就先把这个点在双向链表中删除（因为操作了，后续还要insert进来）
            self.remove(self.my_dict[key])  
            del self.my_dict[key]  # 在哈希中也删除
        if len(self.my_dict) == self.maxLen:  # 如果不在缓存中，并且缓存已经满了
            p = self.R.left      # 那么就要删除掉链表中的最后一个节点
            self.remove(p)
            del self.my_dict[p.val]  # 并且把这个值在哈希表中删除
        p = Node(key, value)   # 新的节点 或者是 被删掉后新的节点
        self.my_dict[key] = p  
        self.insert(p)  # 加入到链表中


    def remove(self, p):
        p.right.left = p.left
        p.left.right = p.right


    def insert(self, p):
        p.right = self.L.right 
        p.left = self.L 
        self.L.right.left = p 
        self.L.right = p
        
        
        
# 法二：用python自带的OrderDict
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity: int):
        self.maxsize = capacity
        self.lrucache = OrderedDict()

    def get(self, key: int) -> int:
        # 说明在缓存中,重新移动字典的尾部
        if key in self.lrucache:
            self.lrucache.move_to_end(key)
        return self.lrucache.get(key, -1)
        
    def put(self, key: int, value: int) -> None:
        # 如果存在,删掉,重新赋值
        if key in self.lrucache:
            del self.lrucache[key]
        # 在字典尾部添加
        self.lrucache[key] = value
        if len(self.lrucache) > self.maxsize:
            # 弹出字典的头部(因为存储空间不够了)
            self.lrucache.popitem(last = False)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 460. LFU 缓存](https://leetcode-cn.com/problems/lfu-cache/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准**

```cpp
// 与lru风格一致的写法
class LFUCache {
public:
    struct Node {
        int k, v, c;
    };

    int cap;
    int min_c;
    unordered_map<int, list<Node>> cache;
    unordered_map<int, list<Node>::iterator> hash;

    Node remove(list<Node>::iterator it) {
        auto [k, v, c] = *it;
        cache[c].erase(it);
        if (cache[c].size() == 0) {
            cache.erase(c);
            if (min_c == c)
                min_c ++ ;
        }
        hash.erase(k);
        return {k, v, c};
    }

    void insert(int k, int v, int c) {
        cache[c].push_front({k, v, c});
        hash[k] = cache[c].begin();
    }

    LFUCache(int capacity) {
        this->cap = capacity;
        this->min_c = 0;
    }
    
    int get(int key) {
        if (hash.find(key) == hash.end())
            return -1;
        auto it = hash[key];
        auto [k, v, c] = remove(it);
        insert(k, v, c + 1);
        return v;
    }
    
    void put(int key, int value) {
        if (cap == 0) return;
        int ori_cnt = 0;
        if (hash.find(key) == hash.end()) {
            if (hash.size() == cap)
                auto _ = remove( -- cache[min_c].end());
        } else {
            auto [_k, _v, _c] = remove(hash[key]);
            ori_cnt = _c;
        }
        if (ori_cnt == 0)
            min_c = 1;
        insert(key, value, ori_cnt + 1);
    }
};

/**
 * Your LFUCache object will be instantiated and called as such:
 * LFUCache* obj = new LFUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
```

##### **C++ 裸链表**

```cpp
// 网络很多做法并不是 O1

class LFUCache {
public:
    struct Node {
        Node *left, *right;
        int key, val;
        Node(int _key, int _val) {
            key = _key, val = _val;
            left = right = NULL;
        }
    };
    struct Block {
        Block *left, *right;
        Node *head, *tail;
        int cnt;
        Block(int _cnt) {
            cnt = _cnt;
            left = right = NULL;
            head = new Node(-1, -1), tail = new Node(-1, -1);
            head->right = tail, tail->left = head;
        }
        ~Block() {
            delete head;
            delete tail;
        }
        void insert(Node* p) {
            p->right = head->right;
            head->right->left = p;
            p->left = head;
            head->right = p;
        }
        void remove(Node* p) {
            p->left->right = p->right;
            p->right->left = p->left;
        }
        bool empty() {
            return head->right == tail;
        }
    }*head, *tail;
    int n;
    unordered_map<int, Block*> hash_block;
    unordered_map<int, Node*> hash_node;

    void insert(Block* p) {  // 在p的右侧插入新块，cnt是p->cnt + 1
        auto cur = new Block(p->cnt + 1);
        cur->right = p->right;
        p->right->left = cur;
        p->right = cur;
        cur->left = p;
    }

    void remove(Block* p) {
        p->left->right = p->right;
        p->right->left = p->left;
        delete p;
    }

    LFUCache(int capacity) {
        n = capacity;
        head = new Block(0), tail = new Block(INT_MAX);
        head->right = tail, tail->left = head;
    }

    int get(int key) {
        if (hash_block.count(key) == 0) return -1;
        auto block = hash_block[key];
        auto node = hash_node[key];
        block->remove(node);
        if (block->right->cnt != block->cnt + 1) insert(block);
        block->right->insert(node);
        hash_block[key] = block->right;
        if (block->empty()) remove(block);
        return node->val;
    }

    void put(int key, int value) {
        if (!n) return;
        if (hash_block.count(key)) {
            hash_node[key]->val = value;
            get(key);
        } else {
            if (hash_block.size() == n) {
                auto p = head->right->tail->left;
                head->right->remove(p);
                if (head->right->empty()) remove(head->right);
                hash_block.erase(p->key);
                hash_node.erase(p->key);
                delete p;
            }
            auto p = new Node(key, value);
            if (head->right->cnt > 1) insert(head);
            head->right->insert(p);
            hash_block[key] = head->right;
            hash_node[key] = p;
        }
    }
};
```

##### **C++ set**

```cpp
struct Node {
    int cnt, time, key, value;
    Node(int c, int t, int k, int v): cnt(c), time(t), key(k), value(v) {}
    bool operator < (const Node& rhs) const {
        return cnt == rhs.cnt ? time < rhs.time : cnt < rhs.cnt;
    }
};

class LFUCache {
public:
    int cap, time;
    unordered_map<int, Node> m;
    set<Node> s;
    LFUCache(int capacity) {
        cap = capacity;
        time = 0;
    }
    
    int get(int key) {
        if (!cap) return -1;
        auto it = m.find(key);
        if (it == m.end()) return -1;
        auto node = it->second;
        s.erase(node);
        ++ node.cnt, node.time = ++ time;
        s.insert(node);
        it->second = node;
        return node.value;
    }
    
    void put(int key, int value) {
        if (!cap) return;
        auto it = m.find(key);
        if (it == m.end()) {
            if (m.size() == cap) {
                m.erase(s.begin()->key);
                s.erase(s.begin());
            }
            Node node = Node(1, ++ time, key, value);
            m.insert({key, node});
            s.insert(node);
        } else {
            Node node = it->second;
            s.erase(node);
            ++ node.cnt, node.time = ++ time, node.value = value;
            s.insert(node);
            it->second = node;
        }
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### 数据结构

> [!NOTE] **[LeetCode 284. 顶端迭代器](https://leetcode-cn.com/problems/peeking-iterator/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/*
 * Below is the interface for Iterator, which is already defined for you.
 * **DO NOT** modify the interface for Iterator.
 *
 *  class Iterator {
 *		struct Data;
 * 		Data* data;
 *		Iterator(const vector<int>& nums);
 * 		Iterator(const Iterator& iter);
 *
 * 		// Returns the next element in the iteration.
 *		int next();
 *
 *		// Returns true if the iteration has more elements.
 *		bool hasNext() const;
 *	};
 */

class PeekingIterator : public Iterator {
public:
    int _next;
    bool _has_next;

    PeekingIterator(const vector<int>& nums) : Iterator(nums) {
        // Initialize any member here.
        // **DO NOT** save a copy of nums and manipulate it directly.
        // You should only use the Iterator interface methods.
        _has_next = Iterator::hasNext();
        if (_has_next)
            _next = Iterator::next();
    }

    // Returns the next element in the iteration without advancing the iterator.
    int peek() {
        return _next;
    }

    // hasNext() and next() should behave the same as in the Iterator interface.
    // Override them if needed.
    int next() {
        int res = _next;
        _has_next = Iterator::hasNext();
        if (_has_next)
            _next = Iterator::next();
        return res;
    }

    bool hasNext() const {
        return _has_next;
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 341. 扁平化嵌套列表迭代器](https://leetcode-cn.com/problems/flatten-nested-list-iterator/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class NestedIterator {
public:
    vector<int> q;
    int k;

    NestedIterator(vector<NestedInteger> &nestedList) {
        k = 0;
        for (auto& l: nestedList) dfs(l);
    }

    void dfs(NestedInteger& l) {
        if (l.isInteger()) q.push_back(l.getInteger());
        else {
            for (auto& v: l.getList()) dfs(v);
        }
    }

    int next() {
        return q[k ++ ];
    }

    bool hasNext() {
        return k < q.size();
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1286. 字母组合迭代器](https://leetcode-cn.com/problems/iterator-for-combination/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 这题复杂度接受直接一次性生成所有可能字符串

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int cur;
    vector<string> v;
    string t;
    int n;
    void dfs(int p, string cur) {
        if (cur.size() == n) {
            v.push_back(cur);
            return;
        }
        if (p == t.size()) return;
        dfs(p + 1, cur);
        dfs(p + 1, cur + t[p]);
    }
    CombinationIterator(string characters, int combinationLength) {
        t = characters;
        n = combinationLength;
        dfs(0, "");
        sort(v.begin(), v.end());
        cur = 0;
    }

    string next() { return v[cur++]; }

    bool hasNext() { return cur < v.size(); }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 380. 常数时间插入、删除和获取随机元素](https://leetcode-cn.com/problems/insert-delete-getrandom-o1/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 设计 trick

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class RandomizedSet {
public:
    /** Initialize your data structure here. */
    unordered_map<int, int> hash;
    vector<int> nums;
    RandomizedSet() {

    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    bool insert(int x) {
        if (hash.count(x) == 0) {
            nums.push_back(x);
            hash[x] = nums.size() - 1;
            return true;
        }
        return false;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    bool remove(int x) {
        if (hash.count(x)) {
            int y = nums.back();
            int px = hash[x], py = hash[y];
            swap(nums[px], nums[py]);
            swap(hash[x], hash[y]);
            hash.erase(x);
            nums.pop_back();
            return true;
        }
        return false;
    }
    
    /** Get a random element from the set. */
    int getRandom() {
        return nums[rand() % nums.size()];
    }
};

/**
 * Your RandomizedSet object will be instantiated and called as such:
 * RandomizedSet* obj = new RandomizedSet();
 * bool param_1 = obj->insert(val);
 * bool param_2 = obj->remove(val);
 * int param_3 = obj->getRandom();
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 381. O(1) 时间插入、删除和获取随机元素 - 允许重复](https://leetcode-cn.com/problems/insert-delete-getrandom-o1-duplicates-allowed/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 设计 trick

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class RandomizedCollection {
public:
    unordered_map<int, unordered_set<int>> hash;
    vector<int> nums;
    /** Initialize your data structure here. */
    RandomizedCollection() {

    }
    
    /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
    bool insert(int val) {
        bool res = hash[val].empty();
        nums.push_back(val);
        hash[val].insert(nums.size() - 1);
        return res;
    }
    
    /** Removes a value from the collection. Returns true if the collection contained the specified element. */
    bool remove(int x) {
        if (hash[x].size()) {
            int px = *hash[x].begin(), py = nums.size() - 1;
            int y = nums.back();
            swap(nums[px], nums[py]);
            hash[x].erase(px), hash[x].insert(py);
            hash[y].erase(py), hash[y].insert(px);
            nums.pop_back();
            hash[x].erase(py);
            return true;
        }
        return false;
    }
    
    /** Get a random element from the collection. */
    int getRandom() {
        return nums[rand() % nums.size()];
    }
};

/**
 * Your RandomizedCollection object will be instantiated and called as such:
 * RandomizedCollection* obj = new RandomizedCollection();
 * bool param_1 = obj->insert(val);
 * bool param_2 = obj->remove(val);
 * int param_3 = obj->getRandom();
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 382. 链表随机节点](https://leetcode-cn.com/problems/linked-list-random-node/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode * h;
    /** @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node. */
    Solution(ListNode* head) {
        h = head;
    }
    
    /** Returns a random node's value. */
    int getRandom() {
        int c = -1, n = 0;
        for (auto p = h; p; p = p->next) {
            ++ n ;
            if (rand() % n == 0) c = p->val;
        }
        return c;
    }
};

/**
 * Your Solution object will be instantiated and called as such:
 * Solution* obj = new Solution(head);
 * int param_1 = obj->getRandom();
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 432. 全 O(1) 的数据结构](https://leetcode-cn.com/problems/all-oone-data-structure/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典 设计题

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class AllOne {
public:
    struct Node {
        Node *left, *right;
        int val;
        unordered_set<string> S;

        Node (int _val) {
            val = _val;
            left = right = NULL;
        }
    }*left, *right;
    unordered_map<string, Node*> hash;

    /** Initialize your data structure here. */
    AllOne() {
        left = new Node(INT_MIN), right = new Node(INT_MAX);
        left->right = right, right->left = left;
    }

    Node* add_to_right(Node* node, string key, int val) {
        if (node->right->val == val) node->right->S.insert(key);
        else {
            auto t = new Node(val);
            t->S.insert(key);
            t->right = node->right, node->right->left = t;
            node->right = t, t->left = node;
        }
        return node->right;
    }

    Node* add_to_left(Node* node, string key, int val) {
        if (node->left->val == val) node->left->S.insert(key);
        else {
            auto t = new Node(val);
            t->S.insert(key);
            t->left = node->left, node->left->right = t;
            node->left = t, t->right = node;
        }
        return node->left;
    }

    void remove(Node* node) {
        node->left->right = node->right;
        node->right->left = node->left;
        delete node;
    }

    /** Inserts a new key <Key> with value 1. Or increments an existing key by 1. */
    void inc(string key) {
        if (hash.count(key) == 0) hash[key] = add_to_right(left, key, 1);
        else {
            auto t = hash[key];
            t->S.erase(key);
            hash[key] = add_to_right(t, key, t->val + 1);
            if (t->S.empty()) remove(t);
        }
    }

    /** Decrements an existing key by 1. If Key's value is 1, remove it from the data structure. */
    void dec(string key) {
        if (hash.count(key) == 0) return;
        auto t = hash[key];
        t->S.erase(key);
        if (t->val > 1) {
            hash[key] = add_to_left(t, key, t->val - 1);
        } else {
            hash.erase(key);
        }
        if (t->S.empty()) remove(t);
    }

    /** Returns one of the keys with maximal value. */
    string getMaxKey() {
        if (right->left != left) return *right->left->S.begin();
        return "";
    }

    /** Returns one of the keys with Minimal value. */
    string getMinKey() {
        if (left->right != right) return *left->right->S.begin();
        return "";
    }
};

/**
 * Your AllOne object will be instantiated and called as such:
 * AllOne* obj = new AllOne();
 * obj->inc(key);
 * obj->dec(key);
 * string param_3 = obj->getMaxKey();
 * string param_4 = obj->getMinKey();
 */
```

##### **Python**

```python

```

##### **Go**

```go
// -------------- Node --------------
type Node struct {
    val int64
    set map[string]struct{}
}

func NewNode(val int64) *Node {
    return &Node {
        val: val,
        set: make(map[string]struct{}),
    }
}

func (n *Node) Erase(key string) {
    delete(n.set, key)
}

func (n *Node) Insert(key string) {
    n.set[key] = struct{}{}
}

func (n *Node) Has(key string) bool {
    _, ok := n.set[key]
    return ok
}

func (n *Node) Size() int64 {
    return int64(len(n.set))
}

func (n *Node) PickOneKey() string {
    for k, _ := range n.set {
        return k
    }
    return ""
}

// -------------- AllOne --------------
type AllOne struct {
    data  *list.List
    hash  map[string]*list.Element
}


func Constructor() AllOne {
    data := list.New()
    data.PushFront(NewNode(math.MinInt64))
    data.PushBack(NewNode(math.MaxInt64))
    return AllOne{
        data: data,
        hash: make(map[string]*list.Element, 0),
    }
}

func (this *AllOne) add_to_right(ele *list.Element, key string, val int64) *list.Element {
    if ele.Next().Value.(*Node).val == val {
        ele.Next().Value.(*Node).Insert(key)
    } else {
        t := NewNode(val)
        t.Insert(key)
        this.data.InsertAfter(t, ele)
    }
    return ele.Next()
}

func (this *AllOne) add_to_left(ele *list.Element, key string, val int64) *list.Element {
    if ele.Prev().Value.(*Node).val == val {
        ele.Prev().Value.(*Node).Insert(key)
    } else {
        t := NewNode(val)
        t.Insert(key)
        this.data.InsertBefore(t, ele)
    }
    return ele.Prev()
}

func (this *AllOne) remove(node *list.Element) {
    this.data.Remove(node)
}


func (this *AllOne) Inc(key string)  {
    if _, ok := this.hash[key]; !ok {
        this.hash[key] = this.add_to_right(this.data.Front(), key, 1)
    } else {
        ele := this.hash[key]
        node := ele.Value.(*Node)
        node.Erase(key)
        this.hash[key] = this.add_to_right(ele, key, node.val + 1)
        if node.Size() == 0 {
            this.remove(ele)
        }
    }
}


func (this *AllOne) Dec(key string)  {
    if _, ok := this.hash[key]; !ok {
        return
    }
    ele := this.hash[key]
    node := ele.Value.(*Node)
    node.Erase(key)
    if node.val > 1 {
        this.hash[key] = this.add_to_left(ele, key, node.val - 1);
    } else {
        delete(this.hash, key)
    }
    if node.Size() == 0 {
        this.remove(ele)
    }
}


func (this *AllOne) GetMaxKey() string {
    if len(this.hash) > 0 {
        return this.data.Back().Prev().Value.(*Node).PickOneKey()
    }
    return ""
}


func (this *AllOne) GetMinKey() string {
    if len(this.hash) > 0 {
        return this.data.Front().Next().Value.(*Node).PickOneKey()
    }
    return ""
}


/**
 * Your AllOne object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Inc(key);
 * obj.Dec(key);
 * param_3 := obj.GetMaxKey();
 * param_4 := obj.GetMinKey();
 */
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 622. 设计循环队列](https://leetcode-cn.com/problems/design-circular-queue/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数组实现 队列与循环队列
> 
> hh与tt的开闭 【前闭后开】
> 
> tt + 1 == hh 主要是为了能够区分队空和队满

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class MyCircularQueue {
public:
    int hh = 0, tt = 0;
    vector<int> q;

    /** Initialize your data structure here. Set the size of the queue to be k. */
    MyCircularQueue(int k) {
        q.resize(k + 1);
    }
    
    /** Insert an element into the circular queue. Return true if the operation is successful. */
    bool enQueue(int value) {
        if (isFull()) return false;
        // ATTENTION 循环队列 tt 开区间
        q[tt ++ ] = value;
        if (tt == q.size()) tt = 0;
        return true;
    }
    
    /** Delete an element from the circular queue. Return true if the operation is successful. */
    bool deQueue() {
        if (isEmpty()) return false;
        ++ hh;
        if (hh == q.size()) hh = 0;
        return true;
    }
    
    /** Get the front item from the queue. */
    int Front() {
        if (isEmpty()) return -1;
        return q[hh];
    }
    
    /** Get the last item from the queue. */
    int Rear() {
        if (isEmpty()) return -1;
        int t = tt - 1;
        if (t < 0) t += q.size();
        return q[t];
    }
    
    /** Checks whether the circular queue is empty or not. */
    bool isEmpty() {
        return hh == tt;
    }
    
    /** Checks whether the circular queue is full or not. */
    bool isFull() {
        return (tt + 1) % q.size() == hh;
    }
};

/**
 * Your MyCircularQueue object will be instantiated and called as such:
 * MyCircularQueue* obj = new MyCircularQueue(k);
 * bool param_1 = obj->enQueue(value);
 * bool param_2 = obj->deQueue();
 * int param_3 = obj->Front();
 * int param_4 = obj->Rear();
 * bool param_5 = obj->isEmpty();
 * bool param_6 = obj->isFull();
 */
```

##### **Python**

```python
"""
1. 用数组模拟 O(1), hh 表示 队列的头， tt 表示 尾
2. 整个队列的有效存储元素的范围是：[hh, t) 【注意：整个过程都是一个  “前闭后开” 区间， 在这个区间内的元素是有效的】
3. 所以 初始状态是 hh == tt = 0, 整个队列其实是空的，因为：[0, 0) 里没有数据
4. 初始化一个队列的最大长度为 k + 1, 这个是为了 用 hh 和 tt 区分 队列的【满】和【空】的状态
5. 定义 hh == tt时：此时队列是空（这里和我们初始化相对应）； (tt + 1) % (k + 1) == hh时，队列是满的
6. 我们也可以用一个额外变量 cnt 来记录队列中元素的个数，这样就队列的最大长度可以为 k。
"""

class MyCircularQueue:

    def __init__(self, k: int):
        # 前闭后开区间，队列是空的
        self.hh, self.tt = 0, 0
        self.q = [0] * (k + 1)

    def enQueue(self, value: int) -> bool:
        if self.isFull():return False 
        # tt 本身是不可大的，所以每次加入数据，就直接把上一轮的 tt 位置赋值就可以，然后再把tt+= 1
        self.q[self.tt] = value
        self.tt += 1
        # 需要做一个边界判断，当 下标 tt == k的时候，越界了，就把 tt 置为0
        if self.tt == len(self.q):
            self.tt = 0 
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():return False
        # 当要取出元素的时候，直接把 队头的指针往后移一位，把那个数移除队列的有效范围呢
        self.hh += 1
        if self.hh == len(self.q):
            self.hh = 0
        return True

    def Front(self) -> int:
        if self.isEmpty():return -1
        return self.q[self.hh]


    def Rear(self) -> int:
        if self.isEmpty():return -1
        t = self.tt - 1
        if t < 0:
            t += len(self.q)
        return self.q[t]


    def isEmpty(self) -> bool:
        return self.hh == self.tt


    def isFull(self) -> bool:
        return (self.tt + 1) % len(self.q) == self.hh



# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 641. 设计循环双端队列](https://leetcode-cn.com/problems/design-circular-deque/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class MyCircularDeque {
public:
    int hh = 0, tt = 0;
    vector<int> q;

    /** Initialize your data structure here. Set the size of the deque to be k. */
    MyCircularDeque(int k) {
        q.resize(k + 1);
    }

    int get(int x) {
        return (x + q.size()) % q.size();
    }
    
    /** Adds an item at the front of Deque. Return true if the operation is successful. */
    bool insertFront(int value) {
        if (isFull()) return false;
        hh = get(hh - 1);
        q[hh] = value;
        return true;
    }
    
    /** Adds an item at the rear of Deque. Return true if the operation is successful. */
    bool insertLast(int value) {
        if (isFull()) return false;
        q[tt ++ ] = value;
        tt = get(tt);
        return true;
    }
    
    /** Deletes an item from the front of Deque. Return true if the operation is successful. */
    bool deleteFront() {
        if (isEmpty()) return false;
        hh = get(hh + 1);
        return true;
    }
    
    /** Deletes an item from the rear of Deque. Return true if the operation is successful. */
    bool deleteLast() {
        if (isEmpty()) return false;
        tt = get(tt - 1);
        return true;
    }
    
    /** Get the front item from the deque. */
    int getFront() {
        if (isEmpty()) return -1;
        return q[hh];
    }
    
    /** Get the last item from the deque. */
    int getRear() {
        if (isEmpty()) return -1;
        return q[get(tt - 1)];
    }
    
    /** Checks whether the circular deque is empty or not. */
    bool isEmpty() {
        return hh == tt;
    }
    
    /** Checks whether the circular deque is full or not. */
    bool isFull() {
        return get(hh - 1) == tt;
    }
};

/**
 * Your MyCircularDeque object will be instantiated and called as such:
 * MyCircularDeque* obj = new MyCircularDeque(k);
 * bool param_1 = obj->insertFront(value);
 * bool param_2 = obj->insertLast(value);
 * bool param_3 = obj->deleteFront();
 * bool param_4 = obj->deleteLast();
 * int param_5 = obj->getFront();
 * int param_6 = obj->getRear();
 * bool param_7 = obj->isEmpty();
 * bool param_8 = obj->isFull();
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 705. 设计哈希集合](https://leetcode-cn.com/problems/design-hashset/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int N = 19997;

class MyHashSet {
public:
    vector<int> h[N];

    /** Initialize your data structure here. */
    MyHashSet() {

    }

    int find(vector<int>& h, int key) {
        for (int i = 0; i < h.size(); i ++ )
            if (h[i] == key)
                return i;
        return -1;
    }

    void add(int key) {
        int t = key % N;
        int k = find(h[t], key);
        if (k == -1) h[t].push_back(key);
    }

    void remove(int key) {
        int t = key % N;
        int k = find(h[t], key);
        if (k != -1) h[t].erase(h[t].begin() + k);
    }

    /** Returns true if this set contains the specified element */
    bool contains(int key) {
        int t = key % N;
        int k = find(h[t], key);
        return k != -1;
    }
};

/**
 * Your MyHashSet object will be instantiated and called as such:
 * MyHashSet* obj = new MyHashSet();
 * obj->add(key);
 * obj->remove(key);
 * bool param_3 = obj->contains(key);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 706. 设计哈希映射](https://leetcode-cn.com/problems/design-hashmap/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int N = 19997;
typedef pair<int, int> PII;

class MyHashMap {
public:
    vector<PII> h[N];

    /** Initialize your data structure here. */
    MyHashMap() {

    }

    int find(vector<PII>& h, int key) {
        for (int i = 0; i < h.size(); i ++ )
            if (h[i].first == key)
                return i;
        return -1;
    }

    /** value will always be non-negative. */
    void put(int key, int value) {
        int t = key % N;
        int k = find(h[t], key);
        if (k == -1) h[t].push_back({key, value});
        else h[t][k].second = value;
    }

    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    int get(int key) {
        int t = key % N;
        int k = find(h[t], key);
        if (k == -1) return -1;
        return h[t][k].second;
    }

    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    void remove(int key) {
        int t = key % N;
        int k = find(h[t], key);
        if (k != -1) h[t].erase(h[t].begin() + k);
    }
};

/**
 * Your MyHashMap object will be instantiated and called as such:
 * MyHashMap* obj = new MyHashMap();
 * obj->put(key,value);
 * int param_2 = obj->get(key);
 * obj->remove(key);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 707. 设计链表](https://leetcode-cn.com/problems/design-linked-list/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class MyLinkedList {
public:
    struct Node {
        int val;
        Node* next;
        Node(int _val): val(_val), next(NULL) {}
    }*head;

    /** Initialize your data structure here. */
    MyLinkedList() {
        head = NULL;
    }

    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    int get(int index) {
        if (index < 0) return -1;
        auto p = head;
        for (int i = 0; i < index && p; i ++ ) p = p->next;
        if (!p) return -1;
        return p->val;
    }

    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    void addAtHead(int val) {
        auto cur = new Node(val);
        cur->next = head;
        head = cur;
    }

    /** Append a node of value val to the last element of the linked list. */
    void addAtTail(int val) {
        if (!head) head = new Node(val);
        else {
            auto p = head;
            while (p->next) p = p->next;
            p->next = new Node(val);
        }
    }

    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    void addAtIndex(int index, int val) {
        if (index <= 0) addAtHead(val);
        else {
            int len = 0;
            for (auto p = head; p; p = p->next) len ++ ;
            if (index == len) addAtTail(val);
            else {
                auto p = head;
                for (int i = 0; i < index - 1; i ++ ) p = p->next;
                auto cur = new Node(val);
                cur->next = p->next;
                p->next = cur;
            }
        }
    }

    /** Delete the index-th node in the linked list, if the index is valid. */
    void deleteAtIndex(int index) {
        int len = 0;
        for (auto p = head; p; p = p->next) len ++ ;
        if (index >= 0 && index < len) {
            if (!index) head = head->next;
            else {
                auto p = head;
                for (int i = 0; i < index - 1; i ++ ) p = p->next;
                p->next = p->next->next;
            }
        }
    }
};

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList* obj = new MyLinkedList();
 * int param_1 = obj->get(index);
 * obj->addAtHead(val);
 * obj->addAtTail(val);
 * obj->addAtIndex(index,val);
 * obj->deleteAtIndex(index);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1146. 快照数组](https://leetcode-cn.com/problems/snapshot-array/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 构造 二分

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class SnapshotArray {
public:
    using PII = pair<int, int>;
    vector<vector<PII>> all;
    vector<int> t;
    int times, snapcnt;
    
    SnapshotArray(int length) {
        times = snapcnt = 0;
        t.clear(); all.resize(length);
        for (int i = 0; i < length; ++ i )
            all[i].push_back({0, 0});
    }
    
    void set(int index, int val) {
        ++ times;
        all[index].push_back({times, val});
    }
    
    int snap() {
        ++ snapcnt;
        t.push_back(times);
        return snapcnt - 1;
    }
    
    int get(int index, int snap_id) {
        int time = t[snap_id];
        auto it = upper_bound(all[index].begin(), all[index].end(), PII{time + 1, -1});
        -- it;
        return it->second;
    }
};

/**
 * Your SnapshotArray object will be instantiated and called as such:
 * SnapshotArray* obj = new SnapshotArray(length);
 * obj->set(index,val);
 * int param_2 = obj->snap();
 * int param_3 = obj->get(index,snap_id);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1166. 设计文件系统](https://leetcode-cn.com/problems/design-file-system/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 模拟 略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class FileSystem {
public:
    unordered_map<string, int> hash;
    
    FileSystem() { }
    
    bool createPath(string path, int value) {
        string fa = path;
        while (fa.back() != '/') fa.pop_back();
        fa.pop_back();
        
        if (fa != "" && !hash.count(fa) || hash.count(path))
            return false;
        hash[path] = value;
        return true;
    }
    
    int get(string path) {
        return hash.count(path) ? hash[path] : -1;
    }
};

/**
 * Your FileSystem object will be instantiated and called as such:
 * FileSystem* obj = new FileSystem();
 * bool param_1 = obj->createPath(path,value);
 * int param_2 = obj->get(path);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1172. 餐盘栈](https://leetcode-cn.com/problems/dinner-plate-stacks/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> BIT 维护即可，注意换静态变量防止超时
> 
> 本质: 快速找到以下数据
> 
> 1. 从左往右第一个未满的栈  push
> 2. 从右往左第一个非空的栈  pop


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 静态 否则TLE
const int N = 100010;
static int tr[N];           // 维护栈中元素数目
static stack<int> stks[N];  // 替换很慢的 vector<stack<int>> stks;

class DinnerPlates {
public:
    // BIT
    void init() {
        memset(tr, 0, sizeof tr);
    }
    int lowbit(int x) {
        return x & -x;
    }
    void add(int x, int v) {
        for (int i = x; i < N; i += lowbit(i)) tr[i] += v;
    }
    int sum(int x) {
        int res = 0;
        for (int i = x; i; i -= lowbit(i)) res += tr[i];
        return res;
    }
    // END OF BIT
    // 左侧第一个未满的栈
    int getL() {
        int l = 1, r = N;
        while (l < r) {
            int m = l + r >> 1;
            if (sum(m) >= m * cap) l = m + 1;
            else r = m;
        }
        return l;
    }
    // 右侧第一个非空的栈
    int getR() {
        int l = 1, r = N;
        while (l < r) {
            int m = l + r >> 1;
            if (sum(m) < tot) l = m + 1;
            else r = m;
        }
        return l;
    }
    
    int cap, tot;
    
    DinnerPlates(int capacity) {
        init();
        cap = capacity; tot = 0;
        // TLE stks = vector<stack<int>>(N);
        for (int i = 0; i < N; ++ i )
            while (!stks[i].empty())
                stks[i].pop();
    }
    
    void push(int val) {
        int idx = getL();
        stks[idx].push(val);
        ++ tot ; add(idx, 1);
    }
    
    int pop() {
        if (!tot) return -1;
        int idx = getR();
        int ret = stks[idx].top(); stks[idx].pop();
        -- tot ; add(idx, -1);
        return ret;
    }
    
    int popAtStack(int index) {
        int idx = index + 1;
        if (stks[idx].empty()) return -1;
        int ret = stks[idx].top(); stks[idx].pop();
        -- tot ; add(idx, -1);
        return ret;
    }
};

/**
 * Your DinnerPlates object will be instantiated and called as such:
 * DinnerPlates* obj = new DinnerPlates(capacity);
 * obj->push(val);
 * int param_2 = obj->pop();
 * int param_3 = obj->popAtStack(index);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1348. 推文计数](https://leetcode-cn.com/problems/tweet-counts-per-frequency/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> stl
> 
> 比赛的时候没有直接申请指定数量的res数组长度 导致实现复杂很多

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class TweetCounts {
public:
    map<string, vector<int>> mp;
    TweetCounts() {}

    void recordTweet(string n, int time) { mp[n].push_back(time); }

    vector<int> getTweetCountsPerFrequency(string freq, string name, int s,
                                           int t) {
        int d;
        if (freq[0] == 'm') d = 60;
        if (freq[0] == 'h') d = 3600;
        if (freq[0] == 'd') d = 24 * 3600;
        int n = (t - s) / d + 1;
        vector<int> ret(n);
        for (int i : mp[name])
            if (s <= i && i <= t) ret[(i - s) / d]++;

        return ret;
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1670. 设计前中后队列](https://leetcode-cn.com/problems/design-front-middle-back-queue/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class FrontMiddleBackQueue {
public:
    // q1.size < q1.size
    deque<int> q1, q2;
    
    FrontMiddleBackQueue() {
        q1.clear(), q2.clear();
    }
    
    void pushFront(int val) {
        q1.push_front(val);
        if (q1.size() >= q2.size()) {
            q2.push_back(q1.back());
            q1.pop_back();
        }
    }
    
    void pushMiddle(int val) {
        if (q1.size() + 2 <= q2.size()) {
            q1.push_back(q2.back());
            q2.pop_back();
        }
        q1.push_back(val);
        if (q1.size() >= q2.size() && q1.size()) {
            q2.push_back(q1.back());
            q1.pop_back();
        }
    }
    
    void pushBack(int val) {
        q2.push_front(val);
        if (q2.size() > q1.size() + 2) {
            q1.push_back(q2.back());
            q2.pop_back();
        }
    }
    
    int popFront() {
        if (q1.empty() && q2.empty()) return -1;
        if (q1.size()) {
            int res = q1.front();
            q1.pop_front();
            if (q2.size() > q1.size() + 2) {
                q1.push_back(q2.back());
                q2.pop_back();
            }
            return res;
        } else {
            int res = q2.back();
            q2.pop_back();
            return res;
        }
    }
    
    int popMiddle() {
        if (q2.empty()) return -1;
        int res = q2.back();
        q2.pop_back();
        if (q1.size() >= q2.size() && q1.size()) {
            q2.push_back(q1.back());
            q1.pop_back();
        }
        return res;
    }
    
    int popBack() {
        if (q2.empty()) return -1;
        int res = q2.front();
        q2.pop_front();
        if (q1.size() >= q2.size() && q1.size()) {
            q2.push_back(q1.back());
            q1.pop_back();
        }
        return res;
    }
};

/**
 * Your FrontMiddleBackQueue object will be instantiated and called as such:
 * FrontMiddleBackQueue* obj = new FrontMiddleBackQueue();
 * obj->pushFront(val);
 * obj->pushMiddle(val);
 * obj->pushBack(val);
 * int param_4 = obj->popFront();
 * int param_5 = obj->popMiddle();
 * int param_6 = obj->popBack();
 */
```

##### **C++ Heltion**

```cpp
class FrontMiddleBackQueue {
public:
    vector<int> v;
    FrontMiddleBackQueue() {

    }
    
    void pushFront(int val) {
        v.insert(v.begin(), val);
    }
    
    void pushMiddle(int val) {
        v.insert(v.begin() + v.size() / 2, val);
    }
    
    void pushBack(int val) {
        v.push_back(val);
    }
    
    int popFront() {
        if(v.empty()) return -1;
        int res = v[0];
        v.erase(v.begin());
        return res;
    }
    
    int popMiddle() {
        if(v.empty()) return -1;
        int res = v[(v.size() - 1) / 2];
        v.erase(v.begin() + (v.size() - 1) / 2);
        return res;
    }
    
    int popBack() {
        if(v.empty()) return -1;
        int res = v.back();
        v.pop_back();
        return res;
    }
};

/**
 * Your FrontMiddleBackQueue object will be instantiated and called as such:
 * FrontMiddleBackQueue* obj = new FrontMiddleBackQueue();
 * obj->pushFront(val);
 * obj->pushMiddle(val);
 * obj->pushBack(val);
 * int param_4 = obj->popFront();
 * int param_5 = obj->popMiddle();
 * int param_6 = obj->popBack();
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 2034. 股票价格波动](https://leetcode-cn.com/problems/stock-price-fluctuation/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 自己两个 map 的直观解法
> 
>   容易看到 对于 hash 来说每次只是用到它的 key 也即价格
> 
>   我们当然可以用一个 multiset 代替 map 来实现统计效果
> 
> - 使用一个 map 和 一个 multiset 的解法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 两个map**

```cpp
class StockPrice {
public:
    map<int, int> hash; // price->count
    map<int, int> data; // ts->price
    
    
    StockPrice() {
        hash.clear();
        data.clear();
    }
    
    void update(int ts, int p) {
        if (data.count(ts)) {
            int ori_p = data[ts];
            hash[ori_p] -- ;
            if (hash[ori_p] == 0)
                hash.erase(ori_p);
        }
        hash[p] ++ ;
        data[ts] = p;
    }
    
    int current() {
        auto it = -- data.end();
        auto [k, v] = *it;
        return v;
    }
    
    int maximum() {
        auto it = -- hash.end();
        auto [k, v] = *it;
        return k;
    }
    
    int minimum() {
        auto it = hash.begin();
        auto [k, v] = *it;
        return k;
    }
};

/**
 * Your StockPrice object will be instantiated and called as such:
 * StockPrice* obj = new StockPrice();
 * obj->update(timestamp,price);
 * int param_2 = obj->current();
 * int param_3 = obj->maximum();
 * int param_4 = obj->minimum();
 */
```

##### **C++ map+multiset**

```cpp
class StockPrice {
    multiset<int> prices;
    map<int, int> history;
    
public:
    StockPrice() {}
    
    void update(int timestamp, int price) {
        if (history.count(timestamp))
            prices.erase(prices.find(history[timestamp]));
        history[timestamp] = price;
        prices.insert(price);
    }
    
    int current() {
        return history.rbegin()->second;
    }
    
    int maximum() {
        return *prices.rbegin();
    }
    
    int minimum() {
        return *prices.begin();
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 优惠活动系统](https://leetcode-cn.com/contest/cnunionpay-2022spring/problems/kDPV0f/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 考虑了用 set 直接存并二分找起始来加速查找
> 
> 实际上暴力遍历就可以过

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class DiscountSystem {
public:
    struct acts {
        int id, p, dis, num, lim;
        unordered_map<int, int> user;
        acts() {}
        acts(int _id, int _p, int _dis, int _num, int _lim) {
            id = _id, p = _p, dis = _dis, num = _num, lim = _lim;
        }
    };
    
    map<int, acts> S;
    
    DiscountSystem() {
        S.clear();
    }
    
    void addActivity(int actId, int priceLimit, int discount, int number, int userLimit) {
        S[actId] = {actId, priceLimit, discount, number, userLimit};
    }
    
    void removeActivity(int actId) {
        S.erase(actId);
    }
    
    int consume(int userId, int cost) {
        int maxDis = 0, id = -1;
        for (auto & [k, v] : S)
            if (v.p <= cost && v.num) {
                auto m = v.user;
                if (m.count(userId) && m[userId] == v.lim)
                    continue;
                if (v.dis > maxDis)
                    maxDis = v.dis, id = k;
            }
        if (id != -1) {
            S[id].user[userId] ++ , S[id].num -- ;
        }
        return cost - maxDis;
    }
};

/**
 * Your DiscountSystem object will be instantiated and called as such:
 * DiscountSystem* obj = new DiscountSystem();
 * obj->addActivity(actId,priceLimit,discount,number,userLimit);
 * obj->removeActivity(actId);
 * int param_3 = obj->consume(userId,cost);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### 四叉树

> [!NOTE] **[LeetCode 427. 建立四叉树](https://leetcode-cn.com/problems/construct-quad-tree/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> s;
    Node* construct(vector<vector<int>>& w) {
        int n = w.size();
        s = vector<vector<int>>(n + 1, vector<int>(n + 1));
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= n; j ++ )
                s[i][j] = s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1] + w[i - 1][j - 1];
        return dfs(1, 1, n, n);
    }

    Node* dfs(int x1, int y1, int x2, int y2) {
        int n = x2 - x1 + 1;
        int sum = s[x2][y2] - s[x2][y1 - 1] - s[x1 - 1][y2] + s[x1 - 1][y1 - 1];
        if (sum == 0 || sum == n * n) return new Node(!!sum, true);
        auto node = new Node(0, false);
        int m = n / 2;
        node->topLeft = dfs(x1, y1, x1 + m - 1, y1 + m - 1);
        node->topRight = dfs(x1, y1 + m, x1 + m - 1, y2);
        node->bottomLeft = dfs(x1 + m, y1, x2, y1 + m - 1);
        node->bottomRight = dfs(x1 + m, y1 + m, x2, y2);
        return node;
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 558. 四叉树交集](https://leetcode-cn.com/problems/logical-or-of-two-binary-grids-represented-as-quad-trees/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/*
// Definition for a QuadTree node.
class Node {
public:
    bool val;
    bool isLeaf;
    Node* topLeft;
    Node* topRight;
    Node* bottomLeft;
    Node* bottomRight;
    
    Node() {
        val = false;
        isLeaf = false;
        topLeft = NULL;
        topRight = NULL;
        bottomLeft = NULL;
        bottomRight = NULL;
    }
    
    Node(bool _val, bool _isLeaf) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = NULL;
        topRight = NULL;
        bottomLeft = NULL;
        bottomRight = NULL;
    }
    
    Node(bool _val, bool _isLeaf, Node* _topLeft, Node* _topRight, Node* _bottomLeft, Node* _bottomRight) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = _topLeft;
        topRight = _topRight;
        bottomLeft = _bottomLeft;
        bottomRight = _bottomRight;
    }
};
*/

class Solution {
public:
    Node* intersect(Node* t1, Node* t2) {
        if (t1->isLeaf) return t1->val ? t1 : t2;
        if (t2->isLeaf) return t2->val ? t2 : t1;
        t1->topLeft = intersect(t1->topLeft, t2->topLeft);
        t1->topRight = intersect(t1->topRight, t2->topRight);
        t1->bottomLeft = intersect(t1->bottomLeft, t2->bottomLeft);
        t1->bottomRight = intersect(t1->bottomRight, t2->bottomRight);

        if (t1->topLeft->isLeaf && t1->topRight->isLeaf && t1->bottomLeft->isLeaf && t1->bottomRight->isLeaf)
            if (t1->topRight->val == t1->topLeft->val
                && t1->bottomLeft->val == t1->topLeft->val
                && t1->bottomRight->val == t1->topLeft->val) {
                    t1->isLeaf = true;
                    t1->val = t1->topLeft->val;
                    t1->topLeft = t1->topRight = t1->bottomLeft = t1->bottomRight = NULL;
                }

        return t1;
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### 随机化

> [!NOTE] **[LeetCode 384. 打乱数组](https://leetcode-cn.com/problems/shuffle-an-array/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> a;

    Solution(vector<int>& nums) {
        a = nums;
    }

    /** Resets the array to its original configuration and return it. */
    vector<int> reset() {
        return a;
    }

    /** Returns a random shuffling of the array. */
    vector<int> shuffle() {
        auto b = a;
        int n = a.size();
        for (int i = 0; i < n; i ++ )
            swap(b[i], b[i + rand() % (n - i)]);
        return b;
    }
};

/**
 * Your Solution object will be instantiated and called as such:
 * Solution* obj = new Solution(nums);
 * vector<int> param_1 = obj->reset();
 * vector<int> param_2 = obj->shuffle();
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### 系统设计

> [!NOTE] **[LeetCode 355. 设计推特](https://leetcode-cn.com/problems/design-twitter/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Twitter {
public:
    /** Initialize your data structure here. */
    typedef pair<int, int> PII;
    #define x first
    #define y second

    unordered_map<int, vector<PII>> tweets;
    unordered_map<int, unordered_set<int>> follows;
    int ts;

    Twitter() {
        ts = 0;
    }
    
    /** Compose a new tweet. */
    void postTweet(int userId, int tweetId) {
        tweets[userId].push_back({ts, tweetId});
        ++ ts;
    }
    
    /** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
    vector<int> getNewsFeed(int userId) {
        priority_queue<vector<int>> heap;
        follows[userId].insert(userId);
        for (auto user : follows[userId]) {
            auto & uts = tweets[user];
            if (uts.size()) {
                int i = uts.size() - 1;
                heap.push({uts[i].first, uts[i].second, i, user});
            }
        }

        vector<int> res;
        for (int i = 0; i < 10 && heap.size(); ++ i ) {
            auto t = heap.top(); heap.pop();
            res.push_back(t[1]);
            int j = t[2];
            if (j) {
                -- j;
                int user = t[3];
                auto & uts = tweets[user];
                heap.push({uts[j].x, uts[j].y, j, user});
            }
        }
        return res;
    }
    
    /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
    void follow(int followerId, int followeeId) {
        follows[followerId].insert(followeeId);
    }
    
    /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
    void unfollow(int followerId, int followeeId) {
        follows[followerId].erase(followeeId);
    }
};

/**
 * Your Twitter object will be instantiated and called as such:
 * Twitter* obj = new Twitter();
 * obj->postTweet(userId,tweetId);
 * vector<int> param_2 = obj->getNewsFeed(userId);
 * obj->follow(followerId,followeeId);
 * obj->unfollow(followerId,followeeId);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 385. 迷你语法分析器](https://leetcode-cn.com/problems/mini-parser/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * class NestedInteger {
 *   public:
 *     // Constructor initializes an empty nested list.
 *     NestedInteger();
 *
 *     // Constructor initializes a single integer.
 *     NestedInteger(int value);
 *
 *     // Return true if this NestedInteger holds a single integer, rather than a nested list.
 *     bool isInteger() const;
 *
 *     // Return the single integer that this NestedInteger holds, if it holds a single integer
 *     // The result is undefined if this NestedInteger holds a nested list
 *     int getInteger() const;
 *
 *     // Set this NestedInteger to hold a single integer.
 *     void setInteger(int value);
 *
 *     // Set this NestedInteger to hold a nested list and adds a nested integer to it.
 *     void add(const NestedInteger &ni);
 *
 *     // Return the nested list that this NestedInteger holds, if it holds a nested list
 *     // The result is undefined if this NestedInteger holds a single integer
 *     const vector<NestedInteger> &getList() const;
 * };
 */
class Solution {
public:
    NestedInteger deserialize(string s) {
        int u = 0;
        return dfs(s, u);
    }

    NestedInteger dfs(string& s, int& u) {
        NestedInteger res;
        if (s[u] == '[') {
            u ++ ;  // 跳过左括号
            while (s[u] != ']') res.add(dfs(s, u));
            u ++ ;  // 跳过右括号
            if (u < s.size() && s[u] == ',') u ++ ;  // 跳过逗号
        } else {
            int k = u;
            while (k < s.size() && s[k] != ',' && s[k] != ']') k ++ ;
            res.setInteger(stoi(s.substr(u, k - u)));
            if (k < s.size() && s[k] == ',') k ++ ;  // 跳过逗号
            u = k;
        }
        return res;
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 535. TinyURL 的加密与解密](https://leetcode-cn.com/problems/encode-and-decode-tinyurl/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    unordered_map<string, string> hash;
    string chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

    string random_str(int k) {
        string res;
        while (k -- ) res += chars[rand() % 62];
        return res;
    }

    // Encodes a URL to a shortened URL.
    string encode(string longUrl) {
        while (true) {
            string shortUrl = random_str(6);
            if (hash.count(shortUrl) == 0) {
                hash[shortUrl] = longUrl;
                return "http://tinyurl.com/" + shortUrl;
            }
        }
        return "";
    }

    // Decodes a shortened URL to its original URL.
    string decode(string shortUrl) {
        return hash[shortUrl.substr(19)];
    }
};

// Your Solution object will be instantiated and called as such:
// Solution solution;
// solution.decode(solution.encode(url));
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1396. 设计地铁系统](https://leetcode-cn.com/problems/design-underground-system/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 主要是建模

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class UndergroundSystem {
    unordered_map<int, pair<string, int>> checkRecord;
    unordered_map<string, pair<double, int>> count;    
    string getStationName(string startStation, string endStation) {
        return startStation + "," + endStation;
    }

public:
    UndergroundSystem() {
    }
    
    void checkIn(int id, string stationName, int t) {
        checkRecord[id] = {stationName, t};
    }
    
    void checkOut(int id, string stationName, int t) {
        string name = getStationName(checkRecord[id].first, stationName);
        t -= checkRecord[id].second;
        count[name].first += (double)t;
        count[name].second += 1;
    }
    
    double getAverageTime(string startStation, string endStation) {
        string name = getStationName(startStation, endStation);
        double ans = count[name].first / (double)count[name].second;
        return ans;
    }
};

```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1472. 设计浏览器历史记录](https://leetcode-cn.com/problems/design-browser-history/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 模拟即可
> 
> 赛榜大多用的数组，自己用栈

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 栈**

```cpp
class Solution {
public:
    stack<string> b, f;
    string now;
    BrowserHistory(string homepage) { this->now = homepage; }

    void visit(string url) {
        b.push(now);
        while (!f.empty()) f.pop();
        now = url;
    }

    string back(int steps) {
        string res = now;
        while (steps && !b.empty()) {
            f.push(res);
            res = b.top();
            b.pop();
            --steps;
        }
        now = res;
        return res;
    }

    string forward(int steps) {
        string res = now;
        while (steps && !f.empty()) {
            b.push(res);
            res = f.top();
            f.pop();
            --steps;
        }
        now = res;
        return res;
    }
};
```

##### **C++ 数组**

```cpp
class Solution {
public:
    int n, m;
    string a[5005];
    BrowserHistory(string homepage) {
        n = m = 0;
        a[0] = homepage;
    }

    void visit(string url) {
        a[n = m = m + 1] = url;
    }

    string back(int steps) {
        steps = min(steps, m);
        m -= steps;
        return a[m];
    }

    string forward(int steps) {
        steps = min(steps, n - m);
        m += steps;
        return a[m];
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1912. 设计电影租借系统](https://leetcode-cn.com/problems/design-movie-rental-system/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 模拟 略
> 
> 注意使用引用

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class MovieRentingSystem {
public:
    using PII = pair<int, int>;
    using TIII = tuple<int, int, int>;
    
    int n;
    vector<vector<int>> es;
    
    unordered_map<int, set<PII>> movies;
    map<PII, int> mprice;
    set<TIII> outs;
    
    MovieRentingSystem(int n, vector<vector<int>>& entries) {
        this->n = n;
        this->es = entries;
        for (auto & e : es) {
            int s = e[0], m = e[1], p = e[2];
            movies[m].insert({p, s});
            mprice[{s, m}] = p;
        }
    }
    
    vector<int> search(int movie) {
        auto s = movies[movie];
        auto it = s.begin();
        vector<int> res;
        for (int i = 0; i < 5 && i < s.size(); ++ i ) {
            res.push_back((*it).second);
            it ++ ;
        }
        return res;
    }
    
    void rent(int shop, int movie) {
        int price = mprice[{shop, movie}];
        auto & s = movies[movie];       // ATTENTION 这里注意加引用... WA1并排错很久
        s.erase({price, shop});
        outs.insert({price, shop, movie});
    }
    
    void drop(int shop, int movie) {
        int price = mprice[{shop, movie}];
        auto & s = movies[movie];
        s.insert({price, shop});
        outs.erase({price, shop, movie});
    }
    
    vector<vector<int>> report() {
        vector<vector<int>> res;
        auto it = outs.begin();
        for (int i = 0; i < 5 && i < outs.size(); ++ i ) {
            auto [p, s, m] = *it;
            res.push_back({s, m});
            it ++ ;
        }
        return res;
    }
};

/**
 * Your MovieRentingSystem object will be instantiated and called as such:
 * MovieRentingSystem* obj = new MovieRentingSystem(n, entries);
 * vector<int> param_1 = obj->search(movie);
 * obj->rent(shop,movie);
 * obj->drop(shop,movie);
 * vector<vector<int>> param_4 = obj->report();
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *