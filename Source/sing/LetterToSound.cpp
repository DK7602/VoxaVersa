// English letter-to-sound rules for out-of-dictionary words, after
// H. S. Elovitz, R. Johnson, A. McHugh, J. E. Shore: "Automatic Translation of English Text to
// Phonetics by Means of Letter-to-Sound Rules", NRL Report 7948, Naval Research Laboratory, 1976
// (a US government work in the public domain). Rewritten for ARPAbet output.
//
// Context symbols:
//   ' '  word boundary          '#'  one or more vowels       ':'  zero or more consonants
//   '^'  one consonant          '.'  one voiced consonant     '+'  one front vowel (E I Y)
//   '%'  a suffix (ER E ES ED ING ELY), right context only
//   '&'  a sibilant (S C G Z X J CH SH)
//   '@'  a consonant that makes a following U "long" without a Y glide (T S R D L Z N J TH CH SH)

#include "G2P.h"

namespace voxa::sing::g2p
{

namespace
{
    struct Rule
    {
        const char* left;
        const char* match;
        const char* right;
        const char* phones;
    };

    const Rule rulesA[] = {
        { "",    "A",     " ",     "AX" },
        { " ",   "ARE",   " ",     "AA R" },
        { " ",   "AR",    "O",     "AX R" },
        { "",    "AR",    "#",     "EH R" },
        { "^",   "AS",    "#",     "EY S" },
        { "",    "A",     "WA",    "AX" },
        { "",    "AW",    "",      "AO" },
        { " :",  "ANY",   "",      "EH N IY" },
        { "",    "A",     "^+#",   "EY" },
        { "#:",  "ALLY",  "",      "AX L IY" },
        { " ",   "AL",    "#",     "AX L" },
        { "",    "AGAIN", "",      "AX G EH N" },
        { "#:",  "AG",    "E",     "IH JH" },
        { "",    "A",     "^+:#",  "AE" },
        { " :",  "A",     "^+ ",   "EY" },
        { "",    "A",     "^%",    "EY" },
        { " ",   "ARR",   "",      "AX R" },
        { "",    "ARR",   "",      "AE R" },
        { " :",  "AR",    " ",     "AA R" },
        { "",    "AR",    " ",     "ER" },
        { "",    "AR",    "",      "AA R" },
        { "",    "AIR",   "",      "EH R" },
        { "",    "AI",    "",      "EY" },
        { "",    "AY",    "",      "EY" },
        { "",    "AU",    "",      "AO" },
        { "#:",  "AL",    " ",     "AX L" },
        { "#:",  "ALS",   " ",     "AX L Z" },
        { "",    "ALK",   "",      "AO K" },
        { "",    "AL",    "^",     "AO L" },
        { " :",  "ABLE",  "",      "EY B AX L" },
        { "",    "ABLE",  "",      "AX B AX L" },
        { "",    "ANG",   "+",     "EY N JH" },
        { "",    "A",     "",      "AE" },
    };

    const Rule rulesB[] = {
        { " ",   "BE",    "^#",    "B IH" },
        { "",    "BEING", "",      "B IY IH NG" },
        { " ",   "BOTH",  " ",     "B OW TH" },
        { " ",   "BUS",   "#",     "B IH Z" },
        { "",    "BUIL",  "",      "B IH L" },
        { "",    "B",     "",      "B" },
    };

    const Rule rulesC[] = {
        { " ",   "CH",    "^",     "K" },
        { "^E",  "CH",    "",      "K" },
        { "",    "CH",    "",      "CH" },
        { " S",  "CI",    "#",     "S AY" },
        { "",    "CI",    "A",     "SH" },
        { "",    "CI",    "O",     "SH" },
        { "",    "CI",    "EN",    "SH" },
        { "",    "C",     "+",     "S" },
        { "",    "CK",    "",      "K" },
        { "",    "COM",   "%",     "K AH M" },
        { "",    "C",     "",      "K" },
    };

    const Rule rulesD[] = {
        { "#:",  "DED",   " ",     "D IH D" },
        { ".E",  "D",     " ",     "D" },
        { "#:^E", "D",    " ",     "T" },
        { " ",   "DE",    "^#",    "D IH" },
        { " ",   "DO",    " ",     "D UW" },
        { " ",   "DOES",  "",      "D AH Z" },
        { " ",   "DOING", "",      "D UW IH NG" },
        { " ",   "DOW",   "",      "D AW" },
        { "",    "DU",    "A",     "JH UW" },
        { "",    "D",     "",      "D" },
    };

    const Rule rulesE[] = {
        { "#:",  "E",     " ",     "" },
        { "':^", "E",     " ",     "" },
        { " :",  "E",     " ",     "IY" },
        { "#",   "ED",    " ",     "D" },
        { "#:",  "E",     "D ",    "" },
        { "",    "EV",    "ER",    "EH V" },
        { "",    "E",     "^%",    "IY" },
        { "",    "ERI",   "#",     "IY R IY" },
        { "",    "ERI",   "",      "EH R IH" },
        { "#:",  "ER",    "#",     "ER" },
        { "",    "ER",    "#",     "EH R" },
        { "",    "ER",    "",      "ER" },
        { " ",   "EVEN",  "",      "IY V EH N" },
        { "#:",  "E",     "W",     "" },
        { "@",   "EW",    "",      "UW" },
        { "",    "EW",    "",      "Y UW" },
        { "",    "E",     "O",     "IY" },
        { "#:&", "ES",    " ",     "IH Z" },
        { "#:",  "E",     "S ",    "" },
        { "#:",  "ELY",   " ",     "L IY" },
        { "#:",  "EMENT", "",      "M EH N T" },
        { "",    "EFUL",  "",      "F UH L" },
        { "",    "EE",    "",      "IY" },
        { "",    "EARN",  "",      "ER N" },
        { " ",   "EAR",   "^",     "ER" },
        { "",    "EAD",   "",      "EH D" },
        { "#:",  "EA",    " ",     "IY AX" },
        { "",    "EA",    "SU",    "EH" },
        { "",    "EA",    "",      "IY" },
        { "",    "EIGH",  "",      "EY" },
        { "",    "EI",    "",      "IY" },
        { " ",   "EYE",   "",      "AY" },
        { "",    "EY",    "",      "IY" },
        { "",    "EU",    "",      "Y UW" },
        { "",    "E",     "",      "EH" },
    };

    const Rule rulesF[] = {
        { "",    "FUL",   "",      "F UH L" },
        { "",    "F",     "",      "F" },
    };

    const Rule rulesG[] = {
        { "",    "GIV",   "",      "G IH V" },
        { " ",   "G",     "I^",    "G" },
        { "",    "GE",    "T",     "G EH" },
        { "SU",  "GGES",  "",      "G JH EH S" },
        { "",    "GG",    "",      "G" },
        { " B#", "G",     "",      "G" },
        { "",    "G",     "+",     "JH" },
        { "",    "GREAT", "",      "G R EY T" },
        { "#",   "GH",    "",      "" },
        { "",    "G",     "",      "G" },
    };

    const Rule rulesH[] = {
        { " ",   "HAV",   "",      "HH AE V" },
        { " ",   "HERE",  "",      "HH IY R" },
        { " ",   "HOUR",  "",      "AW ER" },
        { "",    "HOW",   "",      "HH AW" },
        { "",    "H",     "#",     "HH" },
        { "",    "H",     "",      "" },
    };

    const Rule rulesI[] = {
        { " ",   "IN",    "",      "IH N" },
        { " ",   "I",     " ",     "AY" },
        { "",    "IN",    "D",     "AY N" },
        { "",    "IER",   "",      "IY ER" },
        { "#:R", "IED",   "",      "IY D" },
        { "",    "IED",   " ",     "AY D" },
        { "",    "IEN",   "",      "IY EH N" },
        { "",    "IE",    "T",     "AY EH" },
        { " :",  "I",     "%",     "AY" },
        { "",    "I",     "%",     "IY" },
        { "",    "IE",    "",      "IY" },
        { "",    "I",     "^+:#",  "IH" },
        { "",    "IR",    "#",     "AY R" },
        { "",    "IZ",    "%",     "AY Z" },
        { "",    "IS",    "%",     "AY Z" },
        { "",    "I",     "D%",    "AY" },
        { "+^",  "I",     "^+",    "IH" },
        { "",    "I",     "T%",    "AY" },
        { "#:^", "I",     "^+",    "IH" },
        { "",    "I",     "^+",    "AY" },
        { "",    "IR",    "",      "ER" },
        { "",    "IGH",   "",      "AY" },
        { "",    "ILD",   "",      "AY L D" },
        { "",    "IGN",   " ",     "AY N" },
        { "",    "IGN",   "^",     "AY N" },
        { "",    "IGN",   "%",     "AY N" },
        { "",    "IQUE",  "",      "IY K" },
        { "",    "I",     "",      "IH" },
    };

    const Rule rulesJ[] = {
        { "",    "J",     "",      "JH" },
    };

    const Rule rulesK[] = {
        { " ",   "K",     "N",     "" },
        { "",    "K",     "",      "K" },
    };

    const Rule rulesL[] = {
        { "",    "LO",    "C#",    "L OW" },
        { "L",   "L",     "",      "" },
        { "#:^", "L",     "%",     "AX L" },
        { "",    "LEAD",  "",      "L IY D" },
        { "",    "L",     "",      "L" },
    };

    const Rule rulesM[] = {
        { "",    "MOV",   "",      "M UW V" },
        { "M",   "M",     "",      "" },
        { "",    "M",     "",      "M" },
    };

    const Rule rulesN[] = {
        { "E",   "NG",    "+",     "N JH" },
        { "",    "NG",    "R",     "NG G" },
        { "",    "NG",    "#",     "NG G" },
        { "",    "NGL",   "%",     "NG G AX L" },
        { "",    "NG",    "",      "NG" },
        { "",    "NK",    "",      "NG K" },
        { " ",   "NOW",   " ",     "N AW" },
        { "N",   "N",     "",      "" },
        { "",    "N",     "",      "N" },
    };

    const Rule rulesO[] = {
        { "",    "OF",    " ",     "AX V" },
        { "",    "OROUGH", "",     "ER OW" },
        { "#:",  "OR",    " ",     "ER" },
        { "#:",  "ORS",   " ",     "ER Z" },
        { "",    "OR",    "",      "AO R" },
        { " ",   "ONE",   "",      "W AH N" },
        { "",    "OW",    "",      "OW" },
        { " ",   "OVER",  "",      "OW V ER" },
        { "",    "OV",    "",      "AH V" },
        { "",    "O",     "^%",    "OW" },
        { "",    "O",     "^EN",   "OW" },
        { "",    "O",     "^I#",   "OW" },
        { "",    "OL",    "D",     "OW L" },
        { "",    "OUGHT", "",      "AO T" },
        { "",    "OUGH",  "",      "AH F" },
        { " ",   "OU",    "",      "AW" },
        { "H",   "OU",    "S#",    "AW" },
        { "",    "OUS",   "",      "AX S" },
        { "",    "OUR",   "",      "AO R" },
        { "",    "OULD",  "",      "UH D" },
        { "^",   "OU",    "^L",    "AH" },
        { "",    "OUP",   "",      "UW P" },
        { "",    "OU",    "",      "AW" },
        { "",    "OY",    "",      "OY" },
        { "",    "OING",  "",      "OW IH NG" },
        { "",    "OI",    "",      "OY" },
        { "",    "OOR",   "",      "AO R" },
        { "",    "OOK",   "",      "UH K" },
        { "",    "OOD",   "",      "UH D" },
        { "",    "OO",    "",      "UW" },
        { "",    "O",     "E",     "OW" },
        { "",    "O",     " ",     "OW" },
        { "",    "OA",    "",      "OW" },
        { " ",   "ONLY",  "",      "OW N L IY" },
        { " ",   "ONCE",  "",      "W AH N S" },
        { "",    "ON'T",  "",      "OW N T" },
        { "C",   "O",     "N",     "AA" },
        { "",    "O",     "NG",    "AO" },
        { " :^", "O",     "N",     "AH" },
        { "I",   "ON",    "",      "AX N" },
        { "#:",  "ON",    " ",     "AX N" },
        { "#^",  "ON",    "",      "AX N" },
        { "",    "O",     "ST ",   "OW" },
        { "",    "OF",    "^",     "AO F" },
        { "",    "OTHER", "",      "AH DH ER" },
        { "",    "OSS",   " ",     "AO S" },
        { "#:^", "OM",    "",      "AH M" },
        { "",    "O",     "",      "AA" },
    };

    const Rule rulesP[] = {
        { "",    "PH",    "",      "F" },
        { "",    "PEOP",  "",      "P IY P" },
        { "",    "POW",   "",      "P AW" },
        { "",    "PUT",   " ",     "P UH T" },
        { "P",   "P",     "",      "" },
        { "",    "P",     "",      "P" },
    };

    const Rule rulesQ[] = {
        { "",    "QUAR",  "",      "K W AO R" },
        { "",    "QU",    "",      "K W" },
        { "",    "Q",     "",      "K" },
    };

    const Rule rulesR[] = {
        { " ",   "RE",    "^#",    "R IY" },
        { "R",   "R",     "",      "" },
        { "",    "R",     "",      "R" },
    };

    const Rule rulesS[] = {
        { "",    "SH",    "",      "SH" },
        { "#",   "SION",  "",      "ZH AX N" },
        { "",    "SOME",  "",      "S AH M" },
        { "#",   "SUR",   "#",     "ZH ER" },
        { "",    "SUR",   "#",     "SH ER" },
        { "#",   "SU",    "#",     "ZH UW" },
        { "#",   "SSU",   "#",     "SH UW" },
        { "#",   "SED",   " ",     "Z D" },
        { "#",   "S",     "#",     "Z" },
        { "",    "SAID",  "",      "S EH D" },
        { "^",   "SION",  "",      "SH AX N" },
        { "",    "S",     "S",     "" },
        { ".",   "S",     " ",     "Z" },
        { "#:.E", "S",    " ",     "Z" },
        { "#:^##", "S",   " ",     "Z" },
        { "#:^#", "S",    " ",     "S" },
        { "U",   "S",     " ",     "S" },
        { " :#", "S",     " ",     "Z" },
        { " ",   "SCH",   "",      "S K" },
        { "",    "S",     "C+",    "" },
        { "#",   "SM",    "",      "Z M" },
        { "#",   "SN",    "'",     "Z AX N" },
        { "",    "S",     "",      "S" },
    };

    const Rule rulesT[] = {
        { " ",   "THE",   " ",     "DH AX" },
        { "",    "TO",    " ",     "T UW" },
        { "",    "THAT",  " ",     "DH AE T" },
        { " ",   "THIS",  " ",     "DH IH S" },
        { " ",   "THEY",  "",      "DH EY" },
        { " ",   "THERE", "",      "DH EH R" },
        { "",    "THER",  "",      "DH ER" },
        { "",    "THEIR", "",      "DH EH R" },
        { " ",   "THAN",  " ",     "DH AE N" },
        { " ",   "THEM",  " ",     "DH EH M" },
        { "",    "THESE", " ",     "DH IY Z" },
        { " ",   "THEN",  "",      "DH EH N" },
        { "",    "THROUGH", "",    "TH R UW" },
        { "",    "THOSE", "",      "DH OW Z" },
        { "",    "THOUGH", " ",    "DH OW" },
        { " ",   "THUS",  "",      "DH AH S" },
        { "",    "TH",    "",      "TH" },
        { "#:",  "TED",   " ",     "T IH D" },
        { "S",   "TI",    "#N",    "CH" },
        { "",    "TI",    "O",     "SH" },
        { "",    "TI",    "A",     "SH" },
        { "",    "TIEN",  "",      "SH AX N" },
        { "",    "TUR",   "#",     "CH ER" },
        { "",    "TU",    "A",     "CH UW" },
        { " ",   "TWO",   "",      "T UW" },
        { "T",   "T",     "",      "" },
        { "",    "T",     "",      "T" },
    };

    const Rule rulesU[] = {
        { " ",   "UN",    "I",     "Y UW N" },
        { " ",   "UN",    "",      "AH N" },
        { " ",   "UPON",  "",      "AX P AO N" },
        { "@",   "UR",    "#",     "UH R" },
        { "",    "UR",    "#",     "Y UH R" },
        { "",    "UR",    "",      "ER" },
        { "",    "U",     "^ ",    "AH" },
        { "",    "U",     "^^",    "AH" },
        { "",    "UY",    "",      "AY" },
        { " G",  "U",     "#",     "" },
        { "G",   "U",     "%",     "" },
        { "G",   "U",     "#",     "W" },
        { "#N",  "U",     "",      "Y UW" },
        { "@",   "U",     "",      "UW" },
        { "",    "U",     "",      "Y UW" },
    };

    const Rule rulesV[] = {
        { "",    "VIEW",  "",      "V Y UW" },
        { "",    "V",     "",      "V" },
    };

    const Rule rulesW[] = {
        { " ",   "WERE",  "",      "W ER" },
        { "",    "WA",    "S",     "W AA" },
        { "",    "WA",    "T",     "W AA" },
        { "",    "WHERE", "",      "W EH R" },
        { "",    "WHAT",  "",      "W AA T" },
        { "",    "WHOL",  "",      "HH OW L" },
        { "",    "WHO",   "",      "HH UW" },
        { "",    "WH",    "",      "W" },
        { "",    "WAR",   "",      "W AO R" },
        { "",    "WOR",   "^",     "W ER" },
        { "",    "WR",    "",      "R" },
        { "",    "W",     "",      "W" },
    };

    const Rule rulesX[] = {
        { " ",   "X",     "",      "Z" },
        { "",    "X",     "",      "K S" },
    };

    const Rule rulesY[] = {
        { "",    "YOUNG", "",      "Y AH NG" },
        { " ",   "YOU",   "",      "Y UW" },
        { " ",   "YES",   "",      "Y EH S" },
        { " ",   "Y",     "",      "Y" },
        { "#:^", "Y",     " ",     "IY" },
        { "#:^", "Y",     "I",     "IY" },
        { " :",  "Y",     " ",     "AY" },
        { " :",  "Y",     "#",     "AY" },
        { " :",  "Y",     "^+:#",  "IH" },
        { " :",  "Y",     "^#",    "AY" },
        { "",    "Y",     "",      "IH" },
    };

    const Rule rulesZ[] = {
        { "Z",   "Z",     "",      "" },
        { "",    "Z",     "",      "Z" },
    };

    struct RuleSet { const Rule* rules; int count; };

    template <size_t N>
    constexpr RuleSet set (const Rule (&r)[N]) { return { r, (int) N }; }

    const RuleSet ruleSets[26] = {
        set (rulesA), set (rulesB), set (rulesC), set (rulesD), set (rulesE), set (rulesF), set (rulesG),
        set (rulesH), set (rulesI), set (rulesJ), set (rulesK), set (rulesL), set (rulesM), set (rulesN),
        set (rulesO), set (rulesP), set (rulesQ), set (rulesR), set (rulesS), set (rulesT), set (rulesU),
        set (rulesV), set (rulesW), set (rulesX), set (rulesY), set (rulesZ),
    };

    //==============================================================================
    bool isLetter (char c) noexcept      { return c >= 'A' && c <= 'Z'; }
    bool isVowelLetter (char c) noexcept { return c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U'; }
    bool isConsonantLetter (char c) noexcept { return isLetter (c) && ! isVowelLetter (c); }
    bool isVoicedConsonant (char c) noexcept { return std::strchr ("BDVGJLMNRWZ", c) != nullptr && c != 0; }
    bool isFrontVowel (char c) noexcept  { return c == 'E' || c == 'I' || c == 'Y'; }

    /** Matches a left context right-to-left, ending just before `pos`. */
    bool matchLeft (const char* pattern, const std::string& word, int pos)
    {
        int i = pos;   // index of the character just left of the match
        for (int p = (int) std::strlen (pattern) - 1; p >= 0; --p)
        {
            const char c = pattern[p];
            const char w = i >= 0 ? word[(size_t) i] : ' ';

            if (isLetter (c) || c == '\'')
            {
                if (w != c) return false;
                --i;
            }
            else switch (c)
            {
                case ' ':
                    if (isLetter (w)) return false;
                    --i;
                    break;
                case '#':
                    if (! isVowelLetter (w)) return false;
                    --i;
                    while (i >= 0 && isVowelLetter (word[(size_t) i])) --i;
                    break;
                case ':':
                    while (i >= 0 && isConsonantLetter (word[(size_t) i])) --i;
                    break;
                case '^':
                    if (! isConsonantLetter (w)) return false;
                    --i;
                    break;
                case '.':
                    if (! isVoicedConsonant (w)) return false;
                    --i;
                    break;
                case '+':
                    if (! isFrontVowel (w)) return false;
                    --i;
                    break;
                case '&':
                    if (std::strchr ("SCGZXJ", w) != nullptr && w != 0) { --i; break; }
                    if (i >= 1 && (word[(size_t) i] == 'H') && (word[(size_t) i - 1] == 'C' || word[(size_t) i - 1] == 'S')) { i -= 2; break; }
                    return false;
                case '@':
                    if (std::strchr ("TSRDLZNJ", w) != nullptr && w != 0) { --i; break; }
                    if (i >= 1 && word[(size_t) i] == 'H' && std::strchr ("TCS", word[(size_t) i - 1]) != nullptr) { i -= 2; break; }
                    return false;
                default:
                    return false;
            }
        }
        return true;
    }

    /** Matches a right context left-to-right, starting at `pos`. */
    bool matchRight (const char* pattern, const std::string& word, int pos)
    {
        int i = pos;
        const int n = (int) word.size();
        for (const char* p = pattern; *p != 0; ++p)
        {
            const char c = *p;
            const char w = i < n ? word[(size_t) i] : ' ';

            if (isLetter (c) || c == '\'')
            {
                if (w != c) return false;
                ++i;
            }
            else switch (c)
            {
                case ' ':
                    if (isLetter (w)) return false;
                    ++i;
                    break;
                case '#':
                    if (! isVowelLetter (w)) return false;
                    ++i;
                    while (i < n && isVowelLetter (word[(size_t) i])) ++i;
                    break;
                case ':':
                    while (i < n && isConsonantLetter (word[(size_t) i])) ++i;
                    break;
                case '^':
                    if (! isConsonantLetter (w)) return false;
                    ++i;
                    break;
                case '.':
                    if (! isVoicedConsonant (w)) return false;
                    ++i;
                    break;
                case '+':
                    if (! isFrontVowel (w)) return false;
                    ++i;
                    break;
                case '&':
                    if (i + 1 < n && (w == 'C' || w == 'S') && word[(size_t) i + 1] == 'H') { i += 2; break; }
                    if (std::strchr ("SCGZXJ", w) != nullptr && w != 0) { ++i; break; }
                    return false;
                case '@':
                    if (i + 1 < n && std::strchr ("TCS", w) != nullptr && w != 0 && word[(size_t) i + 1] == 'H') { i += 2; break; }
                    if (std::strchr ("TSRDLZNJ", w) != nullptr && w != 0) { ++i; break; }
                    return false;
                case '%':
                {
                    // ER, E, ES, ED, ING, ELY
                    auto at = [&] (int k) { return i + k < n ? word[(size_t) (i + k)] : ' '; };
                    if (at (0) == 'I' && at (1) == 'N' && at (2) == 'G') { i += 3; break; }
                    if (at (0) != 'E') return false;
                    if (at (1) == 'L' && at (2) == 'Y') { i += 3; break; }
                    if (at (1) == 'R' || at (1) == 'S' || at (1) == 'D') { i += 2; break; }
                    i += 1;
                    break;
                }
                default:
                    return false;
            }
        }
        return true;
    }

    void assignStress (PhoneString& phones)
    {
        int firstVowel = -1, firstFullVowel = -1;
        for (int i = 0; i < (int) phones.size(); ++i)
        {
            auto& p = phones[(size_t) i];
            if (! isVowel (p.ph))
                continue;

            p.stress = 0;
            if (firstVowel < 0)
                firstVowel = i;
            if (firstFullVowel < 0 && p.ph != Ph::AX)
                firstFullVowel = i;
        }

        const int stressed = firstFullVowel >= 0 ? firstFullVowel : firstVowel;
        if (stressed >= 0)
            phones[(size_t) stressed].stress = 1;
    }
}

PhoneString letterToSound (const juce::String& wordIn)
{
    std::string word;
    for (auto c : normaliseWord (wordIn))
    {
        if (c >= 'a' && c <= 'z')
            word += (char) (c - 'a' + 'A');
        else if (c == '\'')
            word += '\'';
    }

    PhoneString phones;
    int pos = 0;
    const int n = (int) word.size();

    while (pos < n)
    {
        const char c = word[(size_t) pos];
        if (! isLetter (c))
        {
            ++pos;
            continue;
        }

        const auto& rs = ruleSets[c - 'A'];
        bool matched = false;

        for (int r = 0; r < rs.count; ++r)
        {
            const auto& rule = rs.rules[r];
            const auto len = (int) std::strlen (rule.match);
            if (pos + len > n || word.compare ((size_t) pos, (size_t) len, rule.match) != 0)
                continue;

            if (! matchLeft (rule.left, word, pos - 1) || ! matchRight (rule.right, word, pos + len))
                continue;

            for (const auto& p : phonesFromString (rule.phones))
            {
                // Collapse doubled consonants produced across rule boundaries ("bookkeeper").
                if (! phones.empty() && ! isVowel (p.ph) && phones.back().ph == p.ph)
                    continue;
                phones.push_back (p);
            }

            pos += len;
            matched = true;
            break;
        }

        if (! matched)
            ++pos;
    }

    // Rules mark every vowel as stressed; replace that with a simple lexical stress pattern.
    for (auto& p : phones)
        if (p.ph == Ph::AH && p.stress != 1)
            p.ph = Ph::AX;

    assignStress (phones);
    return phones;
}

} // namespace voxa::sing::g2p
