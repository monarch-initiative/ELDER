class OMIMHPOExtractor:
    @staticmethod
    def extract_omim_hpo_mappings(data):
        """
        Extracts OMIM to HPO mappings from the provided data.

        :param data: String containing the data with OMIM and HPO information.
        :return: Dictionary with OMIM IDs as keys and lists of HPO IDs as values.
        """
        omim_hpo_dict = {}
        lines = data.split('\n')
        header_skipped = False

        for line in lines:
            if not line or line.startswith('#'):
                continue

            if not header_skipped:
                header_skipped = True
                continue

            parts = line.split('\t')
            if len(parts) < 4:
                continue

            omim_id, disease_name, hpo_id = parts[0], parts[1], parts[3]

            if omim_id in omim_hpo_dict:
                omim_hpo_dict[omim_id].add(hpo_id)
            else:
                omim_hpo_dict[omim_id] = {hpo_id}

        for omim in omim_hpo_dict:
            omim_hpo_dict[omim] = list(omim_hpo_dict[omim])

        return omim_hpo_dict

