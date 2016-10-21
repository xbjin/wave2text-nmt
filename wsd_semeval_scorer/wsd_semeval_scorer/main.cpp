#include <iostream>
#include <stdexcept>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage : <program> <text.tag> <answers.ans>" << endl;
    }

    string text_file_path = argv[1];
    string answers_file_path = argv[2];

    ifstream text_file(text_file_path);
    ifstream answers_file(answers_file_path);

    return 0;
}
