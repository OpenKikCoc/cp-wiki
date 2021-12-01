## 关闭同步/解除绑定

### 代码实现

```cpp
std::ios::sync_with_stdio(false);
std::cin.tie(nullptr);
```

## 读入优化

### 代码实现

```cpp
inline int read() {
    char c = getchar();
    int x = 0, f = 1;
    while (c < '0' || c > '9') {
        if (c == '-')
            f = -1;
        c = getchar();
    }
    while (c >= '0' && c <= '9') {
        x = x * 10 + c - '0';
        c = getchar();
    }
    return x * f;
}
```

## 输出优化

将数字的每一位转化为字符输出以加速

要注意的是，负号要单独判断输出，并且每次 %（mod）取出的是数字末位，因此要倒序输出

### 代码实现

```cpp
void write(int x) {
    if (x < 0) {  // 判负 + 输出负号 + 变原数为正数
        x = -x;
        putchar('-');
    }
    if (x > 9) write(x / 10);  // 递归，将除最后一位外的其他部分放到递归中输出
    putchar(x % 10 + '0');  // 已经输出（递归）完 x 末位前的所有数字，输出末位
}
```

但是递归实现常数是较大的，我们可以写一个栈来实现这个过程

```cpp
inline void write(int x) {
    static int sta[35];
    int top = 0;
    do { sta[top++] = x % 10, x /= 10; } while (x);
    while (top) putchar(sta[--top] + 48);  // 48 是 '0'
}
```

## 输入优化 [Template]

```cpp
template <typename T>
inline T
read() {
    T x = 0, f = 1;
    int ch = getchar();
    for (; !isdigit(ch); ch = getchar())
        if (ch == '-') f = -1;
    for (; isdigit(ch); ch = getchar()) x = x * 10 + ch - '0';
    return x * f;
}
```

如果要分别输入 `int` 类型的变量 a，`long long` 类型的变量 b 和 `__int128` 类型的变量 c，那么可以写成

```cpp
a = read<int>();
b = read<long long>();
c = read<__int128>();
```

## 优化进阶

```cpp
namespace IO {
const int MAXSIZE = 1 << 20;
char buf[MAXSIZE], *p1, *p2;
#define gc()                                                                 \
    (p1 == p2 && (p2 = (p1 = buf) + fread(buf, 1, MAXSIZE, stdin), p1 == p2) \
         ? EOF                                                               \
         : *p1++)
inline int rd() {
    int x = 0, f = 1;
    char c = gc();
    while (!isdigit(c)) {
        if (c == '-') f = -1;
        c = gc();
    }
    while (isdigit(c)) x = x * 10 + (c ^ 48), c = gc();
    return x * f;
}
char pbuf[1 << 20], *pp = pbuf;
inline void push(const char &c) {
    if (pp - pbuf == 1 << 20) fwrite(pbuf, 1, 1 << 20, stdout), pp = pbuf;
    *pp++ = c;
}
inline void write(int x) {
    static int sta[35];
    int top = 0;
    do { sta[top++] = x % 10, x /= 10; } while (x);
    while (top) push(sta[--top] + '0');
}
}  // namespace IO
```

## 优化进阶 [带调试]

关闭调试开关时使用 `fread()`,`fwrite()`，退出时自动析构执行 `fwrite()`。

开启调试开关时使用 `getchar()`,`putchar()`，便于调试。

若要开启文件读写时，请在所有读写之前加入 `freopen()`。

```cpp
// #define DEBUG 1  // 调试开关
struct IO {
#define MAXSIZE (1 << 20)
#define isdigit(x) (x >= '0' && x <= '9')
    char buf[MAXSIZE], *p1, *p2;
    char pbuf[MAXSIZE], *pp;
#if DEBUG
#else
    IO() : p1(buf), p2(buf), pp(pbuf) {}
    ~IO() { fwrite(pbuf, 1, pp - pbuf, stdout); }
#endif
    inline char gc() {
#if DEBUG  // 调试，可显示字符
        return getchar();
#endif
        if (p1 == p2) p2 = (p1 = buf) + fread(buf, 1, MAXSIZE, stdin);
        return p1 == p2 ? ' ' : *p1++;
    }
    inline bool blank(char ch) {
        return ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t';
    }
    template <class T>
    inline void read(T &x) {
        register double tmp = 1;
        register bool sign = 0;
        x = 0;
        register char ch = gc();
        for (; !isdigit(ch); ch = gc())
            if (ch == '-') sign = 1;
        for (; isdigit(ch); ch = gc()) x = x * 10 + (ch - '0');
        if (ch == '.')
            for (ch = gc(); isdigit(ch); ch = gc())
                tmp /= 10.0, x += tmp * (ch - '0');
        if (sign) x = -x;
    }
    inline void read(char *s) {
        register char ch = gc();
        for (; blank(ch); ch = gc())
            ;
        for (; !blank(ch); ch = gc()) *s++ = ch;
        *s = 0;
    }
    inline void read(char &c) {
        for (c = gc(); blank(c); c = gc())
            ;
    }
    inline void push(const char &c) {
#if DEBUG  // 调试，可显示字符
        putchar(c);
#else
        if (pp - pbuf == MAXSIZE) fwrite(pbuf, 1, MAXSIZE, stdout), pp = pbuf;
        *pp++ = c;
#endif
    }
    template <class T>
    inline void write(T x) {
        if (x < 0) x = -x, push('-');  // 负数输出
        static T sta[35];
        T top = 0;
        do { sta[top++] = x % 10, x /= 10; } while (x);
        while (top) push(sta[--top] + '0');
    }
    template <class T>
    inline void write(T x, char lastChar) {
        write(x), push(lastChar);
    }
} io;
```

## 输入输出的缓冲

`printf` 和 `scanf` 是有缓冲区的。这也就是为什么，如果输入函数紧跟在输出函数之后/输出函数紧跟在输入函数之后可能导致错误。

### 刷新缓冲区

1. 程序结束
2. 关闭文件
3. `printf` 输出 `\r` 或者 `\n` 到终端的时候（注：如果是输出到文件，则不会刷新缓冲区）
4. 手动 `fflush()`
5. 缓冲区满自动刷新
6. `cout` 输出 `endl`