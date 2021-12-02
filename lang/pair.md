
> [!NOTE] **更多元素的关联可以使用 [tuple](lang/new#stdtuple)**

## 使用

### 初始化

```cpp
pair<int, double> p0(1, 2.0);
```

```cpp
pair<int, double> p2 = make_pair(1, 2.0);
```

在 C++11 以及之后

```cpp
auto p3 = make_pair(1, 2.0);
```

### 访问 & 修改

```cpp
int i = p0.first;
double d = p0.second;
```

```cpp
p1.first++;
```

### 比较

`pair` 已经预先定义了所有的比较运算符，包括 `<`、`>`、`<=`、`>=`、`==`、`!=`。当然，这需要组成 `pair` 的两个变量所属的数据类型定义了 `==` 和/或 `<` 运算符。

其中，`<`、`>`、`<=`、`>=` 四个运算符会先比较两个 `pair` 中的第一个变量，在第一个变量相等的情况下再比较第二个变量。

```cpp
if (p2 >= p3) {
    cout << **do something here" << endl;
}
```

由于 `pair` 定义了 STL 中常用的 `<` 与 `==`，使得其能够很好的与其他 STL 函数或数据结构配合。比如，`pair` 可以作为 `priority_queue` 的数据类型。

```cpp
priority_queue<pair<int, double> > q;
```

### 赋值与交换

```cpp
p0 = p1;
```

```cpp
swap(p0, p1);
p2.swap(p3);
```

## 应用举例

### 离散化

将原始数据的值作为每个 `pair` 第一个变量，将原始数据的位置作为第二个变量。

在排序后，将原始数据值的排名（该值排序后所在的位置）赋给该值原本所在的位置即可。

```cpp
// a为原始数据
pair<int, int> a[MAXN];
// ai为离散化后的数据
int ai[MAXN];
for (int i = 0; i < n; i++) {
    // first为原始数据的值，second为原始数据的位置
    scanf("%d", &a[i].first);
    a[i].second = i;
}
// 排序
sort(a, a + n);
for (int i = 0; i < n; i++) {
    // 将该值的排名赋给该值原本所在的位置
    ai[a[i].second] = i;
}
```

### Dijkstra

```cpp
using PII = pair<int, int>;

priority_queue<PII, std::vector<PII>, std::greater<PII> > q;
...
while (!q.empty()) {
    // dis为入堆时节点到起点的距离，i为节点编号
    auto [dis, i] = q.top();
    q.pop();
    ...
}
```

### pair 与 map

`map` 的是 C++ 中存储键值对的数据结构。很多情况下，`map` 中存储的键值对通过 `pair` 向外暴露。

```cpp
map<int, double> m;
m.insert(make_pair(1, 2.0));
```

> [!NOTE] **unordered_map 不可直接使用 PII 作为 key [需要计算 hash]**
> 
> 如需使用 PII 作 key 需 map