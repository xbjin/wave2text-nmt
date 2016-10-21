#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage : <program> <hyp.tag> <ref.tag>" << endl;
        return -1;
    }

    string hyp_file_path = argv[1];
    string ref_file_path = argv[2];

    ifstream hyp_file(hyp_file_path);
    ifstream ref_file(ref_file_path);

    string ref_word;
    string hyp_word;

    int match = 0;

    bool found = false;

    string ref_line;
    string hyp_line;

    while (getline(ref_file, ref_line))
    {
        istringstream ref_line_stream(ref_line);
        if (!getline(hyp_file, hyp_line)) break;
        istringstream hyp_line_stream(hyp_line);
        while (ref_line_stream >> ref_word)
        {
            auto ref_lemma_sense_delimiter_position = ref_word.find_first_of('%');
            string ref_lemma = ref_word.substr(0, ref_lemma_sense_delimiter_position);
            string ref_sense = "";
            if (ref_lemma_sense_delimiter_position == ref_word.npos)
            {
                ref_sense = "";
            }
            else
            {
                ref_sense = ref_word.substr(ref_lemma_sense_delimiter_position);
            }
            auto position = hyp_line_stream.tellg();
            found = false;
            while (!found && hyp_line_stream >> hyp_word)
            {
                auto hyp_lemma_sense_delimiter_position = hyp_word.find_first_of('%');
                string hyp_lemma = hyp_word.substr(0, hyp_lemma_sense_delimiter_position);
                string hyp_sense = "";
                if (hyp_lemma_sense_delimiter_position == hyp_word.npos)
                {
                    hyp_sense = "";
                }
                else
                {
                    hyp_sense = hyp_word.substr(hyp_lemma_sense_delimiter_position);
                }
                if (hyp_lemma == ref_lemma)
                {
                    if (ref_sense != "" && hyp_sense == ref_sense)
                    {
                        match += 1;
                    }
                    found = true;
                }
            }
            if (!found)
            {
                hyp_file.seekg(position);
            }
        }
    }

    cout << match;
    cout.flush();

    return 0;
}
