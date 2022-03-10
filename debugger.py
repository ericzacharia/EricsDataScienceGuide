def countBinarySubstrings(s: str) -> int:
    pointer_1 = 0
    pointer_2 = 1
    length_a = 1
    length_b = 0
    count = 0
    sub_strings = []  # keeps track of the sub strings
    while pointer_2 < len(s):
        if s[pointer_1] == s[pointer_2]:
            length_a += 1
            pointer_2 += 1
        else:
            for j in range(pointer_2, len(s)):
                if s[j] != s[pointer_1]:
                    length_b += 1
                else:
                    length_a = 1
                    length_b = 0
                    pointer_1 += 1
                    pointer_2 = pointer_1 + 1
                    break
                if length_b == length_a:  # found a sub string
                    sub_strings.append(s[pointer_1:j+1])
                    count += 1
                    length_a = 1
                    length_b = 0
                    pointer_1 += 1
                    pointer_2 = pointer_1 + 1
                    break
                elif length_b > length_a:
                    length_a = 1
                    length_b = 0
                    pointer_1 += 1
                    pointer_2 = pointer_1 + 1
                    break

    return count, sub_strings


# print(countBinarySubstrings("00100"))
# print(countBinarySubstrings("10101"))
# countBinarySubstrings("00110011")
countBinarySubstrings("00110")
