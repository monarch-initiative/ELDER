import json
import unittest
from pheval_elder.prepare.core.OMIMHPOExtractor import \
    OMIMHPOExtractor
from pheval_elder.prepare.core.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.data_processor import DataProcessor


class TestOMIMHPOExtractor(unittest.TestCase):

    def test_extract_omim_hpo_mappings_with_frequencies(self):
        sample_data = """
        #description: "HPO annotations for rare diseases [8181: OMIM; 47: DECIPHER; 4242 ORPHANET]"
        #version: 2023-10-09
        #tracker: https://github.com/obophenotype/human-phenotype-ontology/issues
        #hpo-version: http://purl.obolibrary.org/obo/hp/releases/2023-10-09/hp.json
        database_id	disease_name	qualifier	hpo_id	reference	evidence	onset	frequency	sex	modifier	aspect	biocuration
        OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0011097	PMID:31675180	PCS		1/2			P	HPO:probinson[2021-06-21]
        OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0002187	PMID:31675180	PCS		1/1			P	HPO:probinson[2021-06-21]
        """

        expected_output = {"OMIM:619340": {"HP:0011097": 0.5,"HP:0002187": 1.0}}
        result = OMIMHPOExtractor.extract_omim_hpo_mappings_with_frequencies_1(sample_data)
        self.assertEqual(result, expected_output)

    def test_extract_omim_hpo_mappings_default(self):
        sample_data = '''
        #description: "HPO annotations for rare diseases [8181: OMIM; 47: DECIPHER; 4242 ORPHANET]"
        #version: 2023-10-09
        #tracker: https://github.com/obophenotype/human-phenotype-ontology/issues
        #hpo-version: http://purl.obolibrary.org/obo/hp/releases/2023-10-09/hp.json
        database_id	disease_name	qualifier	hpo_id	reference	evidence	onset	frequency	sex	modifier	aspect	biocuration
        OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0011097	PMID:31675180	PCS		1/2			P	HPO:probinson[2021-06-21]
        OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0002187	PMID:31675180	PCS		1/1			P	HPO:probinson[2021-06-21]
        '''

        expected_output = {
            "OMIM:619340": [
                "HP:0011097",
                "HP:0002187"
            ],
        }

        result = OMIMHPOExtractor.extract_omim_hpo_mappings_default(sample_data)
        print(result)

        self.assertEqual(result, expected_output)


    def test_file_read_to_pretty_json(self):
        file_path = "/Users/carlo/PycharmProjects/chroma_db_playground/phenotypeTestFile.hpoa"
        data = OMIMHPOExtractor.read_data_from_file(file_path)
        obj = OMIMHPOExtractor.extract_omim_hpo_mappings_with_frequencies_1(data)
        fname = "omim_dict_test.json"
        OMIMHPOExtractor.save_results_as_pretty_json_string(obj, fname)
        test_file_path = "/Users/carlo/Downloads/pheval.exomiser/output/" + fname
        result = json.loads(OMIMHPOExtractor.read_data_from_file(test_file_path))

        expected = json.loads("""
        {
            "OMIM:609153": {
                "HP:0000006": 0.5,
                "HP:0002153": 0.5,
                "HP:0002378": 0.5,
                "HP:0003324": 0.5,
                "HP:0003394": 0.5,
                "HP:0003768": 0.5
            },
            "OMIM:610370": {
                "HP:0000007": 0.5,
                "HP:0001508": 0.5,
                "HP:0001944": 0.5,
                "HP:0002013": 0.5,
                "HP:0002014": 0.5,
                "HP:0003623": 0.5,
                "HP:0004918": 0.5
            },
            "OMIM:614102": {
                "HP:0000007": 0.5,
                "HP:0002014": 0.2,
                "HP:0002028": 1.0,
                "HP:0002205": 1.0,
                "HP:0002719": 1.0,
                "HP:0010701": 0.5,
                "HP:0011463": 1.0
            },
            "OMIM:619340": {
                "HP:0000006": 0.5,
                "HP:0001518": 0.5,
                "HP:0001522": 0.5,
                "HP:0001789": 0.5,
                "HP:0002187": 1.0,
                "HP:0002643": 1.0,
                "HP:0010851": 1.0,
                "HP:0011097": 0.5,
                "HP:0011451": 0.5,
                "HP:0032792": 0.5,
                "HP:0200134": 1.0
            },
            "OMIM:619426": {
                "HP:0000006": 0.5,
                "HP:0000072": 0.125,
                "HP:0000085": 0.25,
                "HP:0000126": 0.125,
                "HP:0000143": 0.125,
                "HP:0000154": 0.125,
                "HP:0000219": 0.25,
                "HP:0000278": 0.25,
                "HP:0000286": 0.375,
                "HP:0000293": 0.625,
                "HP:0000369": 0.125,
                "HP:0000400": 0.625,
                "HP:0000403": 0.25,
                "HP:0000430": 0.25,
                "HP:0000463": 0.125,
                "HP:0000506": 0.25,
                "HP:0000527": 0.25,
                "HP:0000537": 0.25,
                "HP:0000574": 0.25,
                "HP:0000582": 0.125,
                "HP:0000601": 0.125,
                "HP:0000637": 0.125,
                "HP:0000639": 0.125,
                "HP:0000664": 0.25,
                "HP:0000739": 0.125,
                "HP:0000821": 0.125,
                "HP:0001249": 1.0,
                "HP:0001252": 0.875,
                "HP:0001385": 0.125,
                "HP:0001388": 0.5,
                "HP:0001513": 0.25,
                "HP:0001545": 0.25,
                "HP:0002020": 0.125,
                "HP:0002870": 0.125,
                "HP:0003196": 0.5,
                "HP:0003593": 0.625,
                "HP:0003621": 0.25,
                "HP:0005280": 0.125,
                "HP:0006989": 0.125,
                "HP:0007018": 0.125,
                "HP:0010804": 0.125,
                "HP:0011228": 0.375,
                "HP:0011330": 0.125,
                "HP:0011463": 0.125,
                "HP:0011800": 0.25,
                "HP:0012745": 0.125,
                "HP:0020045": 0.125,
                "HP:0020206": 0.125,
                "HP:0034003": 0.25
            }
        }
        """)

        self.assertEqual(result, expected)

    def test_file_read_to_pretty_json_with_default_extractor(self):
        file_path = "/Users/carlo/PycharmProjects/chroma_db_playground/phenotypeTestFile.hpoa"
        data = OMIMHPOExtractor.read_data_from_file(file_path)
        obj = OMIMHPOExtractor.extract_omim_hpo_mappings_default(data)
        fname = "omim_dict_default_extract_test.json"
        OMIMHPOExtractor.save_results_as_pretty_json_string(obj, fname)
        test_file_path = "/Users/carlo/Downloads/pheval.exomiser/output/" + fname
        result = json.loads(OMIMHPOExtractor.read_data_from_file(test_file_path))
        expected = json.loads("""
            {
                "OMIM:609153": [
                    "HP:0001878",
                    "HP:0003394",
                    "HP:0003768",
                    "HP:0002378",
                    "HP:0002153",
                    "HP:0003324",
                    "HP:0000006"
                ],
                "OMIM:610370": [
                    "HP:0004918",
                    "HP:0000007",
                    "HP:0003623",
                    "HP:0001508",
                    "HP:0002013",
                    "HP:0002014",
                    "HP:0001944"
                ],
                "OMIM:614102": [
                    "HP:0000007",
                    "HP:0002205",
                    "HP:0002719",
                    "HP:0002028",
                    "HP:0010701",
                    "HP:0002014",
                    "HP:0011463"
                ],
                "OMIM:619340": [
                    "HP:0002187",
                    "HP:0001518",
                    "HP:0032792",
                    "HP:0001789",
                    "HP:0002643",
                    "HP:0010851",
                    "HP:0001522",
                    "HP:0200134",
                    "HP:0011451",
                    "HP:0011097",
                    "HP:0000006"
                ],
                "OMIM:619426": [
                    "HP:0000400",
                    "HP:0000506",
                    "HP:0000739",
                    "HP:0000637",
                    "HP:0000369",
                    "HP:0011330",
                    "HP:0000143",
                    "HP:0000527",
                    "HP:0034003",
                    "HP:0020206",
                    "HP:0012745",
                    "HP:0000219",
                    "HP:0000278",
                    "HP:0001385",
                    "HP:0000154",
                    "HP:0000072",
                    "HP:0003621",
                    "HP:0010804",
                    "HP:0003196",
                    "HP:0000126",
                    "HP:0011228",
                    "HP:0000463",
                    "HP:0002020",
                    "HP:0011463",
                    "HP:0001252",
                    "HP:0001545",
                    "HP:0003593",
                    "HP:0007018",
                    "HP:0000537",
                    "HP:0001513",
                    "HP:0000286",
                    "HP:0000403",
                    "HP:0000664",
                    "HP:0000821",
                    "HP:0001249",
                    "HP:0002870",
                    "HP:0011800",
                    "HP:0000582",
                    "HP:0000601",
                    "HP:0000006",
                    "HP:0000574",
                    "HP:0001388",
                    "HP:0000639",
                    "HP:0000293",
                    "HP:0006989",
                    "HP:0005280",
                    "HP:0000085",
                    "HP:0000430",
                    "HP:0020045"
                ]
            }
            """)

        # Sort the lists within each OMIM ID for both result and expected
        for key in result:
            result[key].sort()
        for key in expected:
            expected[key].sort()

        self.assertEqual(result, expected)



if __name__ == '__main__':
    unittest.main()
