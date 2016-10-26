#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

bool is_synset(string token)
{
    if (token.length() != 9) return false;
    char pos = token[0];
    if (pos != 'n' && pos != 'v' && pos != 'a' && pos != 'r') return false;
    for (size_t i = 1 ; i < 9 ; i++)
    {
        if (!isdigit(token[i])) return false;
    }
    return true;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage : <program> <hyp.syn> <ref.syn>" << endl;
        return -1;
    }

    string hyp_file_path = argv[1];
    string ref_file_path = argv[2];

    ifstream hyp_file(hyp_file_path);
    ifstream ref_file(ref_file_path);

    string hyp_line;
    string ref_line;

    string hyp_word;
    string ref_word;

    int match = 0;

    bool found = false;

    while (getline(ref_file, ref_line))
    {
        istringstream ref_line_stream(ref_line);
        if (!getline(hyp_file, hyp_line)) break;
        istringstream hyp_line_stream(hyp_line);
        while (ref_line_stream >> ref_word)
        {
            auto position = hyp_line_stream.tellg();
            found = false;
            while (!found && hyp_line_stream >> hyp_word)
            {
                if (hyp_word == ref_word)
                {
                    found = true;
                    if (is_synset(ref_word))
                    {
                        match += 1;
                    }
                }
            }
            if (!found)
            {
                hyp_line_stream.seekg(position);
            }
        }
    }

    cout << match;
    cout.flush();

    return 0;
}
