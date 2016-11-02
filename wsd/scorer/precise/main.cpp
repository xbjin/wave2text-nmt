#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

using namespace std;

string extract_lemma(string token)
{
    auto lemma_sense_delimiter_position = token.find_first_of('%');
    string lemma = token.substr(0, lemma_sense_delimiter_position);
    return lemma;
}

string extract_sense(string token)
{
    auto lemma_sense_delimiter_position = token.find_first_of('%');
    string sense = "";
    if (lemma_sense_delimiter_position == token.npos)
    {
        sense = "";
    }
    else if (lemma_sense_delimiter_position + 1 == token.npos)
    {
        sense = "";
    }
    else
    {
        sense = token.substr(lemma_sense_delimiter_position + 1);
    }
    return sense;
}

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

    string hyp_line;
    string ref_line;

    string hyp_word;
    string ref_word;

    int total = 0;
    int match = 0;

    bool found = false;

    while (getline(ref_file, ref_line))
    {
        istringstream ref_line_stream(ref_line);
        if (!getline(hyp_file, hyp_line)) break;
        istringstream hyp_line_stream(hyp_line);
        while (ref_line_stream >> ref_word)
        {
            string ref_lemma = extract_lemma(ref_word);
            string ref_sense = extract_sense(ref_word);
            auto position = hyp_line_stream.tellg();
            found = false;
            while (!found && hyp_line_stream >> hyp_word)
            {
                string hyp_lemma = extract_lemma(hyp_word);
                string hyp_sense = extract_sense(hyp_word);
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
                hyp_line_stream.clear();
                hyp_line_stream.seekg(position);
            }
            if (ref_sense != "")
            {
                total += 1;
            }
        }
    }

    float precision = static_cast<float>(match) / static_cast<float>(total);
    precision *= 100.f;

    cout << fixed << setprecision(2) << precision;
    cout.flush();

    return 0;
}
