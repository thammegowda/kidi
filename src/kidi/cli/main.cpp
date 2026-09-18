#include <filesystem>
#include <iostream>
#include <string_view>

#include "kidi/core/version.h"
#include "kidi/rtg/package.h"
#include "kidi/rtg/translator.h"

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
        std::cout << "kidi " << kidi::core::version() << '\n';
        return 0;
    }

    if (argc == 3 && std::string_view(argv[1]) == "inspect") {
        auto package = kidi::rtg::Package::load(std::filesystem::path(argv[2]));
        if (!package) {
            std::cerr << "kidi: " << package.error().message << '\n';
            return 1;
        }
        const auto& manifest = package->manifest();
        std::cout << "format: " << manifest.format_version << '\n'
                  << "model: " << manifest.model_type << '\n'
                  << "weights: " << manifest.weights_file.string() << '\n'
                  << "weight tensors: " << package->weights().size() << '\n'
                  << "vocabularies: " << package->source_tokenizer().vocabulary_size() << ", "
                  << package->target_tokenizer().vocabulary_size() << '\n'
                  << "beam: " << manifest.decode_defaults.beam_size << '\n';
        return 0;
    }

    if (argc == 4 && std::string_view(argv[1]) == "translate") {
        auto translator = kidi::rtg::Translator::load(std::filesystem::path(argv[2]));
        if (!translator) {
            std::cerr << "kidi: " << translator.error().message << '\n';
            return 1;
        }
        auto translation = translator->translate(argv[3]);
        if (!translation) {
            std::cerr << "kidi: " << translation.error().message << '\n';
            return 1;
        }
        std::cout << translation->text << '\n';
        return 0;
    }

    std::cerr << "usage: kidi --version\n"
              << "       kidi inspect MODEL_YAML\n"
              << "       kidi translate MODEL_YAML TEXT\n";
    return 2;
}